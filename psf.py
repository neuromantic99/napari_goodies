from os import mkdir
from typing import Annotated, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage.measure import centroid
from models import PrincipleAxisGaussian, FWHM, StackRange
from pathlib import Path

HERE = Path(__file__).parent


def get_estimates(data: np.ndarray, spacing: Annotated[List[float], 3]) -> List:
    max_ = data.max()
    mean_ = data.mean()
    cz, cy, cx = centroid(data)
    cz *= spacing[0]
    cy *= spacing[1]
    cx *= spacing[2]
    cov = get_cov_matrix(data, spacing)
    return [
        max_ - mean_,
        mean_,
        cx,
        cy,
        cz,
        cov[0, 0],
        cov[0, 1],
        cov[0, 2],
        cov[1, 1],
        cov[1, 2],
        cov[2, 2],
    ]


def fit_gaussian_3d(
    data_cropped: np.ndarray, spacing: Annotated[List[float], 3]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    inspired from napari_psf_analysis.

    data_cropped: 3d array (cropped beadstack)
    spacing: 1d, 3-length array-like (units)
    """

    zz = np.arange(data_cropped.shape[0]) * spacing[0]
    yy = np.arange(data_cropped.shape[1]) * spacing[1]
    xx = np.arange(data_cropped.shape[2]) * spacing[2]
    z, y, x = np.meshgrid(zz, yy, xx, indexing="ij")
    coords = np.stack([z.ravel(), y.ravel(), x.ravel()], -1)
    p0_estimates = get_estimates(data_cropped, spacing)
    print(p0_estimates)

    popt, pcov = curve_fit(ellipsoid3D, coords, data_cropped.ravel(), p0=p0_estimates)

    return popt, pcov


def ellipsoid3D(
    data: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mu_x: int,
    mu_y: int,
    mu_z: int,
    cxx: int,
    cxy: int,
    cxz: int,
    cyy: int,
    cyz: int,
    czz: int,
) -> np.ndarray:
    inv = np.linalg.inv(
        np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
        + np.identity(3) * 1e-6
    )
    return (
        A
        * np.exp(
            -0.5
            * (
                inv[0, 0] * (data[:, 2] - mu_x) ** 2
                + 2 * inv[0, 1] * (data[:, 2] - mu_x) * (data[:, 1] - mu_y)
                + 2 * inv[0, 2] * (data[:, 2] - mu_x) * (data[:, 0] - mu_z)
                + inv[1, 1] * (data[:, 1] - mu_y) ** 2
                + 2 * inv[1, 2] * (data[:, 1] - mu_y) * (data[:, 0] - mu_z)
                + inv[2, 2] * (data[:, 0] - mu_z) ** 2
            )
        )
        + B
    )


def cov(x: np.ndarray, y: np.ndarray, i: np.ndarray) -> float:
    return np.sum(x * y * i) / np.sum(i)


def get_cov_matrix(img: np.ndarray, spacing: Annotated[List[float], 3]) -> np.ndarray:

    z, y, x = np.meshgrid(
        np.arange(img.shape[0]) * spacing[0],
        np.arange(img.shape[1]) * spacing[1],
        np.arange(img.shape[2]) * spacing[2],
        indexing="ij",
    )
    cen = np.asarray(
        [
            np.round(img.shape[0] / 2),
            np.round(img.shape[1] / 2),
            np.round(img.shape[2] / 2),
        ]
    )
    z = z.ravel() - cen[0] * spacing[0]
    y = y.ravel() - cen[1] * spacing[1]
    x = x.ravel() - cen[2] * spacing[2]

    cxx = cov(x, x, img.ravel())
    cyy = cov(y, y, img.ravel())
    czz = cov(z, z, img.ravel())
    cxy = cov(x, y, img.ravel())
    cxz = cov(x, z, img.ravel())
    cyz = cov(y, z, img.ravel())

    return np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])


def compute_fwhm(popt: np.ndarray) -> Tuple[FWHM, PrincipleAxisGaussian, np.ndarray]:
    height = popt[0]
    background = popt[1]
    cxx = popt[5]
    cxy = popt[6]
    cxz = popt[7]
    cyy = popt[8]
    cyz = popt[9]
    czz = popt[10]
    cov = np.array([[cxx, cxy, cxz], [cxy, cyy, cyz], [cxz, cyz, czz]])
    eigval, eigvec = np.linalg.eig(cov)
    pa3, pa2, pa1 = np.sort(np.sqrt(eigval)) / 2
    return (
        FWHM(
            x=2 * np.sqrt(2 * np.log(2)) * np.sqrt(cxx),
            y=2 * np.sqrt(2 * np.log(2)) * np.sqrt(cyy),
            z=2 * np.sqrt(2 * np.log(2)) * np.sqrt(czz),
        ),
        PrincipleAxisGaussian(x=pa1, y=pa2, z=pa3),
        eigvec,
    )


def plot_psf(
    popt: np.ndarray,
    fwhm: FWHM,
    stack_range: StackRange,
    px_size_um: float,
    step_size_um: float,
    principle_axis: PrincipleAxisGaussian,
    eigvec: np.ndarray,
) -> None:
    mu_x = popt[2]
    mu_y = popt[3]
    mu_z = popt[4]

    annot_colour = "blue"
    scalebar_size_um = 1

    xy_spacing = 1
    z_spacing = 1

    fig, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=300)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    fontprops = fm.FontProperties(size=12)
    cmap = "Greys_r"

    axes[0, 0].imshow(stack_range.xy_maxZ, cmap=cmap)
    axes[0, 0].set_xlabel("x")
    axes[0, 0].xaxis.set_label_position("top")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].yaxis.set_label_position("left")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    scalebar = AnchoredSizeBar(
        axes[0, 0].transData,
        scalebar_size_um / px_size_um,
        "1 um",
        "lower right",
        pad=0.1,
        color=annot_colour,
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops,
    )
    axes[0, 0].add_artist(scalebar)

    cx = mu_x / xy_spacing + 0.5
    cy = mu_y / xy_spacing + 0.5
    cz = (mu_z + z_spacing * 0.5) / xy_spacing + 0.5

    dx = (fwhm.x / 2) / xy_spacing
    dy = (fwhm.y / 2) / xy_spacing
    dz = (fwhm.z / 2) / z_spacing

    axes[0, 0].plot(
        [cx - dx, cx + dx],
        [
            stack_range.y_range / 2,
        ]
        * 2,
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[0, 0].plot(
        [cx - dx, cx - dx], [cy - dy, stack_range.y_range / 2], ":", c=annot_colour
    )
    axes[0, 0].plot(
        [cx + dx, cx + dx], [cy - dy, stack_range.y_range / 2], ":", c=annot_colour
    )

    axes[0, 0].plot(
        [
            stack_range.x_range / 2,
        ]
        * 2,
        [cy - dy, cy + dy],
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[0, 0].plot(
        [stack_range.x_range / 2, cx - dx], [cy - dy, cy - dy], ":", c=annot_colour
    )
    axes[0, 0].plot(
        [stack_range.x_range / 2, cx - dx], [cy + dy, cy + dy], ":", c=annot_colour
    )

    axes[1, 0].imshow(stack_range.xz_maxY, cmap=cmap)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].xaxis.set_label_position("bottom")
    axes[1, 0].set_ylabel("z")
    axes[1, 0].yaxis.set_label_position("left")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    scalebar = AnchoredSizeBar(
        axes[1, 0].transData,
        scalebar_size_um / px_size_um,
        "1 um",
        "lower right",
        pad=0.1,
        color=annot_colour,
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops,
    )
    axes[1, 0].add_artist(scalebar)

    axes[1, 0].plot(
        [cx - dx, cx + dx],
        [
            stack_range.z_range / 2,
        ]
        * 2,
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[1, 0].plot(
        [cx - dx, cx - dx], [cz - dz, stack_range.z_range / 2], ":", c=annot_colour
    )
    axes[1, 0].plot(
        [cx + dx, cx + dx], [cz - dz, stack_range.z_range / 2], ":", c=annot_colour
    )

    axes[1, 0].plot(
        [
            stack_range.x_range / 2,
        ]
        * 2,
        [cz - dz, cz + dz],
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[1, 0].plot(
        [stack_range.x_range / 2, cx - dx], [cz - dz, cz - dz], ":", c=annot_colour
    )
    axes[1, 0].plot(
        [stack_range.x_range / 2, cx - dx], [cz + dz, cz + dz], ":", c=annot_colour
    )

    axes[0, 1].imshow(stack_range.yz_maxX.T, cmap=cmap)
    axes[0, 1].set_xlabel("z")
    axes[0, 1].xaxis.set_label_position("top")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].yaxis.set_label_position("right")
    axes[0, 1].yaxis.label.set_rotation(-90)
    axes[0, 1].yaxis.labelpad = 12
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    scalebar = AnchoredSizeBar(
        axes[0, 1].transData,
        scalebar_size_um / step_size_um,
        "1 um",
        "lower right",
        pad=0.1,
        color=annot_colour,
        frameon=False,
        size_vertical=1,
        fontproperties=fontprops,
    )
    axes[0, 1].add_artist(scalebar)

    axes[0, 1].plot(
        [cz - dz, cz + dz],
        [
            stack_range.y_range / 2,
        ]
        * 2,
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[0, 1].plot(
        [cz - dz, cz - dz], [cy - dy, stack_range.y_range / 2], ":", c=annot_colour
    )
    axes[0, 1].plot(
        [cz + dz, cz + dz], [cy - dy, stack_range.y_range / 2], ":", c=annot_colour
    )

    axes[0, 1].plot(
        [
            stack_range.z_range / 2,
        ]
        * 2,
        [cy - dy, cy + dy],
        linewidth=4,
        c=annot_colour,
        solid_capstyle="butt",
    )
    axes[0, 1].plot(
        [stack_range.z_range / 2, cz - dz], [cy - dy, cy - dy], ":", c=annot_colour
    )
    axes[0, 1].plot(
        [stack_range.z_range / 2, cz - dz], [cy + dy, cy + dy], ":", c=annot_colour
    )

    bbox_size = principle_axis.x // 200 + 1.5
    axes[1, 1] = fig.add_axes(
        (0.525, 0.025, 0.45, 0.45), projection="3d", computed_zorder=False
    )
    axes[1, 1].set_xlim(-bbox_size, bbox_size)
    axes[1, 1].set_ylim(-bbox_size, bbox_size)
    axes[1, 1].set_zlim(-bbox_size, bbox_size)

    # Hide grid lines
    axes[1, 1].grid(False)

    # Hide axes ticks
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_zticks([])

    neg = -eigvec[:, 0] * principle_axis.x / 2.0
    pos = eigvec[:, 0] * principle_axis.x / 2.0
    axes[1, 1].quiver3D(
        0, 0, 0, *neg, linewidth=1, zorder=2, arrow_length_ratio=0, color="#0061B5"
    )
    axes[1, 1].quiver3D(
        0, 0, 0, *pos, linewidth=1, zorder=2, arrow_length_ratio=0, color="#0061B5"
    )
    neg = -eigvec[:, 1] * principle_axis.y / 2.0
    pos = eigvec[:, 1] * principle_axis.y / 2.0
    axes[1, 1].quiver3D(
        0, 0, 0, *neg, linewidth=1, zorder=2, arrow_length_ratio=0, color="#D81B60"
    )
    axes[1, 1].quiver3D(
        0, 0, 0, *pos, linewidth=1, zorder=2, arrow_length_ratio=0, color="#D81B60"
    )
    neg = -eigvec[:, 2] * principle_axis.z / 2.0
    pos = eigvec[:, 2] * principle_axis.z / 2.0
    axes[1, 1].quiver3D(
        0, 0, 0, *neg, linewidth=1, zorder=2, arrow_length_ratio=0, color="#03A919"
    )
    axes[1, 1].quiver3D(
        0, 0, 0, *pos, linewidth=1, zorder=2, arrow_length_ratio=0, color="#03A919"
    )

    axes[1, 1].view_init(30, 60)

    axes[1, 1].set_xlabel("x", labelpad=-15)
    axes[1, 1].set_ylabel("y", labelpad=-15)
    axes[1, 1].set_zlabel("z", labelpad=-15)

    if not (HERE / "figures").exists():
        mkdir(HERE / "figures")

    plt.savefig(HERE / "figures" / "2024-05-31_PsfSize.pdf", dpi=300)
