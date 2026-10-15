"""Orion : diametres INTRALUMINAUX de l'aorte ascendante.

A lancer APRES orion_cavites_glb.py (il lit sa segmentation), depuis
l'environnement monai-physio :

    python orion_diametre_aorte.py

CONVENTION : DIAMETRE INTRALUMINAL (lumiere opacifiee), PAS de bord externe a
bord externe comme le demandent la plupart des recommandations en scanner. La
paroi (environ 2 a 3 mm) n'est pas comptee : la valeur est donc PLUS PETITE
que la mesure de reference, en attendant le recalage avec un radiologue.

Trois segmentations de l'aorte sont comparees, toutes sans modele sous licence :

* ``monai`` : label 7 de ``results/orion_cavites.labelmap.mha`` (bundle MONAI
  ``wholeBody_ct_segmentation``, Apache 2.0, 1,5 mm).
* ``densite`` : affinage deterministe sur le scanner injecte. Seuil a
  mi-hauteur entre la lumiere (mediane du coeur du masque MONAI) et le
  myocarde (mediane du label 44), croissance a partir de ce coeur, bornee a
  6 mm autour de l'ASCENDANTE MONAI et hors des autres cavites et de l'artere
  pulmonaire (pas de fuite dans le ventricule). Trous combles.
* ``totalsegmentator`` : tache OUVERTE ``total`` de TotalSegmentator, pleine
  resolution, si ``results/orion_ts_total/total_aorte.nii.gz`` existe
  (voir results/orion_ts_total/lancer.py).

Mesure (identique pour les trois) :

1. Aorte ASCENDANTE : composante connexe la plus proche du ventricule gauche.
2. Reechantillonnage isotrope (0,6 mm), axe central par squelettisation
   (``totalsegmentator.aorta_report.centerline.get_centerline``, Apache 2.0),
   lisse.
3. Pour chaque point de l'axe, tous les 2 mm : section PERPENDICULAIRE a l'axe,
   echantillonnee en millimetres (grille 0,2 mm), blob contenant l'axe. Trois
   diametres : maximal et minimal (largeurs de Feret extremes), equivalent a
   l'aire (2 * sqrt(A / pi)). Validation : tests/test_orion_diametre_aorte.py
   (tubes synthetiques, erreur <= 0,5 mm).
4. Ecartes : 10 premiers et derniers mm de l'axe, et toute section qui touche
   le bord du champ scanne (vaisseau tronque, sous-estime).

CE QUE CA NE DIT PAS
- Reperes de la racine (anneau, sinus, jonction sino-tubulaire) : taches
  TotalSegmentator sous licence, NON localises ici.
- Scanner public (Slicer-Heart-CT), pas un patient Orion ; toute valeur reste
  a confirmer par un radiologue.

Sorties : ``results/orion_diametre_aorte.json`` (profil par source) et
``results/orion_diametre_aorte.png`` (image de controle avant / apres).
"""

from __future__ import annotations

import json
from pathlib import Path

import itk
import numpy as np
from scipy import ndimage

RESULTATS = Path(__file__).resolve().parent / "results"
LABEL_AORTE = 7
LABEL_VENTRICULE_GAUCHE = 46  # wholeBody_ct_segmentation (TotalSegmentator v1)
PAS_ISOTROPE_MM = 0.6
PAS_SECTION_MM = 0.2
DEMI_SECTION_MM = 35.0
MARGE_EXTREMITES_MM = 10.0
PAS_AXE_MM = 2.0
LISSAGE_MM = 0.6


def isotrope(masque: np.ndarray, espacement: np.ndarray, pas: float) -> np.ndarray:
    """Masque booleen reechantillonne a `pas` mm dans chaque direction."""
    facteurs = espacement / pas
    return ndimage.zoom(masque.astype(np.float32), facteurs, order=1) >= 0.5


def axe_central(masque: np.ndarray, pas: float) -> np.ndarray:
    """Points de l'axe (mm, ordre des axes du tableau), lisses et reechantillonnes."""
    from totalsegmentator.aorta_report.centerline import get_centerline

    _, chemin = get_centerline(masque)
    points = np.asarray([v.point for v in chemin], dtype=float) * pas
    lisses = ndimage.gaussian_filter1d(points, sigma=4.0, axis=0, mode="nearest")
    longueurs = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(lisses, axis=0), axis=1)))
    )
    abscisses = np.arange(0.0, longueurs[-1], PAS_AXE_MM)
    return np.stack(
        [np.interp(abscisses, longueurs, lisses[:, k]) for k in range(3)], axis=1
    )


def _base_orthonormee(tangente: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = tangente / np.linalg.norm(tangente)
    reference = (
        np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(t, reference)
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def _diametres_extremes(points: np.ndarray) -> tuple[float, float]:
    """(maximal, minimal) d'un nuage 2D en mm : largeurs de Feret extremes.

    Le minimal est la plus petite largeur toutes directions confondues, et non
    la largeur perpendiculaire au maximal : sur une section ovale, plusieurs
    cordes ont presque la longueur maximale et la direction retenue bascule,
    ce qui gonflerait une largeur dite perpendiculaire.
    """
    from scipy.spatial import ConvexHull

    sommets = points[ConvexHull(points).vertices]
    distances = np.linalg.norm(sommets[:, None, :] - sommets[None, :, :], axis=2)
    angles = np.radians(np.arange(0.0, 180.0, 0.25))
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    projections = sommets @ directions.T
    largeurs = projections.max(axis=0) - projections.min(axis=0)
    return float(distances.max()), float(largeurs.min())


def section(
    masque_f: np.ndarray, pas: float, centre: np.ndarray, tangente: np.ndarray
) -> dict | None:
    """Diametres (mm) de la section perpendiculaire a l'axe en `centre` (mm)."""
    u, v = _base_orthonormee(tangente)
    grille = np.arange(-DEMI_SECTION_MM, DEMI_SECTION_MM + 1e-9, PAS_SECTION_MM)
    a, b = np.meshgrid(grille, grille, indexing="ij")
    coords_mm = centre[:, None, None] + u[:, None, None] * a + v[:, None, None] * b
    indices = coords_mm / pas
    valeurs = ndimage.map_coordinates(masque_f, indices, order=1, mode="constant")
    plan = valeurs >= 0.5
    # Points du plan hors du volume scanne : le masque y vaut 0 par defaut,
    # ce qui tronquerait le vaisseau sans que rien ne le signale.
    forme = np.asarray(masque_f.shape)[:, None, None]
    hors_champ = np.any((indices < 0) | (indices > forme - 1), axis=0)
    etiquettes, _ = ndimage.label(plan)
    milieu = len(grille) // 2
    blob = etiquettes[milieu, milieu]
    if blob == 0:
        return None
    region = etiquettes == blob
    # Section ouverte sur le bord de la grille, ou touchant le bord du champ du
    # scanner : vaisseau tronque, ecartee plutot que sous-estimee.
    if (
        region[0, :].any()
        or region[-1, :].any()
        or region[:, 0].any()
        or region[:, -1].any()
    ):
        return None
    if (ndimage.binary_dilation(region, iterations=3) & hors_champ).any():
        return None
    # Largeurs mesurees sur le CONTOUR a mi-hauteur (interpolation sous-pixel),
    # pas sur les centres des cellules : ces derniers raccourciraient chaque
    # largeur d'environ un pas de grille.
    from skimage.measure import find_contours

    voisinage = ndimage.binary_dilation(region, iterations=2)
    contours = find_contours(np.where(voisinage, valeurs, 0.0), 0.5)
    contour = max(contours, key=len)
    points = grille[0] + contour * PAS_SECTION_MM
    maximal, minimal = _diametres_extremes(points)
    aire = float(region.sum()) * PAS_SECTION_MM**2
    return {
        "diametre_max_mm": round(maximal, 1),
        "diametre_min_mm": round(minimal, 1),
        "diametre_aire_mm": round(2.0 * np.sqrt(aire / np.pi), 1),
    }


def profil(masque_iso: np.ndarray, pas: float) -> list[dict]:
    axe = axe_central(masque_iso, pas)
    # Lissage leger (sigma 0,6 mm) avant le seuil a 0,5 : il efface l'escalier
    # des voxels d'origine (+/- 0,3 mm) qui gonflait le diametre maximal et
    # creusait le minimal, sans deplacer un bord plat ; sur un vaisseau de
    # rayon R, le bord recule d'environ sigma^2 / 2R (0,015 mm pour R = 12 mm).
    masque_f = ndimage.gaussian_filter(
        masque_iso.astype(np.float32), sigma=LISSAGE_MM / pas
    )
    longueur = (len(axe) - 1) * PAS_AXE_MM
    lignes = []
    for k in range(len(axe)):
        s = k * PAS_AXE_MM
        if s < MARGE_EXTREMITES_MM or s > longueur - MARGE_EXTREMITES_MM:
            continue
        tangente = axe[min(k + 2, len(axe) - 1)] - axe[max(k - 2, 0)]
        mesure = section(masque_f, pas, axe[k], tangente)
        if mesure:
            lignes.append({"abscisse_mm": round(s, 1), **mesure})
    return lignes


CAVITES_ET_ARTERE_PULMONAIRE = (45, 46, 47, 48, 49)
LABEL_MYOCARDE = 44
MARGE_DENSITE_MM = 6.0
CONVENTION = "intraluminal"
AVERTISSEMENT = (
    "Diametre INTRALUMINAL (lumiere opacifiee), pas de bord externe a bord externe : "
    "plus petit que la mesure de reference d'environ l'epaisseur de la paroi. "
    "Scanner public, pas un patient Orion ; reperes de racine non localises ; "
    "a confirmer par un radiologue."
)


def aorte_ascendante(aorte: np.ndarray, ventricule_gauche: np.ndarray) -> np.ndarray:
    """Composante connexe significative de l'aorte la plus proche du ventricule gauche."""
    composantes, n = ndimage.label(aorte)
    if n == 0:
        raise SystemExit("Aucune aorte dans la segmentation.")
    vg = np.argwhere(ventricule_gauche)
    if len(vg) == 0:
        raise SystemExit(
            "Ventricule gauche absent : aorte ascendante non identifiable."
        )
    centre_vg = vg.mean(axis=0)
    tailles = ndimage.sum(np.ones_like(composantes), composantes, range(1, n + 1))
    # Les ilots de quelques voxels sont du bruit de segmentation.
    candidates = [i + 1 for i, t in enumerate(tailles) if t > 0.05 * tailles.max()]
    distances = {
        c: np.min(
            np.linalg.norm(np.argwhere(composantes == c)[::50] - centre_vg, axis=1)
        )
        for c in candidates
    }
    return composantes == min(distances, key=distances.get)


def masque_densite(
    labels: np.ndarray, ct: np.ndarray, espacement: np.ndarray, monai: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Aorte ascendante affinee : seuil a mi-hauteur lumiere / myocarde, croissance
    a partir du coeur de `monai` (l'ascendante MONAI), bornee a son voisinage --
    sinon la croissance rejoint l'aorte descendante par la crosse."""
    coeur = ndimage.binary_erosion(monai, iterations=2)
    lumiere = float(np.median(ct[coeur]))
    myocarde = float(
        np.median(ct[ndimage.binary_erosion(labels == LABEL_MYOCARDE, iterations=1)])
    )
    seuil = 0.5 * (lumiere + myocarde)
    rayon = np.maximum(1, np.round(MARGE_DENSITE_MM / espacement)).astype(int)
    boule = np.zeros(2 * rayon + 1, bool)
    zz, yy, xx = np.indices(boule.shape) - rayon[:, None, None, None]
    boule = (zz * espacement[0]) ** 2 + (yy * espacement[1]) ** 2 + (
        xx * espacement[2]
    ) ** 2 <= MARGE_DENSITE_MM**2
    autorise = ndimage.binary_dilation(monai, structure=boule) & ~np.isin(
        labels, CAVITES_ET_ARTERE_PULMONAIRE
    )
    candidats = (ct >= seuil) & autorise
    composantes, _ = ndimage.label(candidats)
    graines = np.unique(composantes[coeur & candidats])
    affine = np.isin(composantes, graines[graines > 0])
    affine = ndimage.binary_fill_holes(affine)
    return affine, {
        "lumiere_hu": round(lumiere),
        "myocarde_hu": round(myocarde),
        "seuil_hu": round(seuil),
    }


def masque_totalsegmentator(reference) -> tuple[np.ndarray, np.ndarray] | None:
    """(aorte, coeur) de la tache ouverte `total`, remis sur la grille de reference."""
    chemin = RESULTATS / "orion_ts_total" / "total_aorte.nii.gz"
    if not chemin.exists():
        return None
    from totalsegmentator.map_to_binary import class_map

    noms = {v: k for k, v in class_map["total"].items()}
    image = itk.imread(str(chemin))
    resample = itk.ResampleImageFilter.New(
        Input=image,
        ReferenceImage=reference,
        UseReferenceImage=True,
        Interpolator=itk.NearestNeighborInterpolateImageFunction.New(image),
    )
    resample.Update()
    labels = itk.array_from_image(resample.GetOutput())
    return labels == noms["aorta"], labels == noms["heart"]


def mesurer(ascendante: np.ndarray, espacement: np.ndarray) -> dict:
    iso = isotrope(ascendante, espacement, PAS_ISOTROPE_MM)
    lignes = profil(iso, PAS_ISOTROPE_MM)
    if not lignes:
        return {"profil": [], "maximum": None, "longueur_mesuree_mm": 0.0}
    return {
        "longueur_mesuree_mm": lignes[-1]["abscisse_mm"] - lignes[0]["abscisse_mm"],
        "maximum": max(lignes, key=lambda sec: sec["diametre_max_mm"]),
        "maximum_aire": max(lignes, key=lambda sec: sec["diametre_aire_mm"]),
        "profil": lignes,
    }


def image_de_controle(
    sources: dict, ct: np.ndarray, espacement: np.ndarray, chemin: Path
) -> None:
    """Une ligne par source : profil, puis quatre sections avec le contour sur le scanner."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct_iso = ndimage.zoom(ct, espacement / PAS_ISOTROPE_MM, order=1)
    grille = np.arange(-DEMI_SECTION_MM, DEMI_SECTION_MM + 1e-9, PAS_SECTION_MM)
    a, b = np.meshgrid(grille, grille, indexing="ij")
    fig, axes = plt.subplots(
        len(sources), 5, figsize=(22, 4.4 * len(sources)), squeeze=False
    )
    for ligne, (nom, (masque, mesure)) in enumerate(sources.items()):
        iso = isotrope(masque, espacement, PAS_ISOTROPE_MM)
        axe = axe_central(iso, PAS_ISOTROPE_MM)
        ax = axes[ligne]
        s = [sec["abscisse_mm"] for sec in mesure["profil"]]
        for cle, libelle in (
            ("diametre_max_mm", "max (Feret)"),
            ("diametre_min_mm", "minimal"),
            ("diametre_aire_mm", "equivalent aire"),
        ):
            ax[0].plot(s, [sec[cle] for sec in mesure["profil"]], label=libelle)
        ax[0].set_title(f"{nom} : profil intraluminal")
        ax[0].set_xlabel("abscisse depuis la racine (mm)")
        ax[0].set_ylabel("mm")
        ax[0].set_ylim(20, 50)
        ax[0].legend(fontsize=8)
        for i, cible in enumerate((12.0, 22.0, 40.0, 60.0)):
            k = min(int(round(cible / PAS_AXE_MM)), len(axe) - 1)
            u, v = _base_orthonormee(axe[min(k + 2, len(axe) - 1)] - axe[max(k - 2, 0)])
            coords = (
                axe[k][:, None, None] + u[:, None, None] * a + v[:, None, None] * b
            ) / PAS_ISOTROPE_MM
            ax[i + 1].imshow(
                np.clip(ndimage.map_coordinates(ct_iso, coords, order=1), -200, 700),
                cmap="gray",
            )
            ax[i + 1].contour(
                ndimage.map_coordinates(iso.astype(np.float32), coords, order=1),
                [0.5],
                colors="r",
                linewidths=1,
            )
            sec = next((x for x in mesure["profil"] if x["abscisse_mm"] == cible), None)
            ax[i + 1].set_title(
                f"{nom} {cible:.0f} mm : "
                + (
                    f"max {sec['diametre_max_mm']} / aire {sec['diametre_aire_mm']}"
                    if sec
                    else "ecartee"
                )
            )
            ax[i + 1].axis("off")
    plt.tight_layout()
    plt.savefig(chemin, dpi=70)
    plt.close(fig)


def main() -> None:
    reference = itk.imread(str(RESULTATS / "orion_cavites.labelmap.mha"))
    labels = itk.array_from_image(reference)
    ct = itk.array_from_image(itk.imread(str(RESULTATS / "slice_fixed.mha"))).astype(
        np.float32
    )
    espacement = np.asarray(
        tuple(reference.GetSpacing())[::-1], dtype=float
    )  # (z, y, x)
    vg = labels == LABEL_VENTRICULE_GAUCHE

    sources: dict[str, tuple[np.ndarray, dict]] = {}
    details: dict[str, dict] = {}
    monai = aorte_ascendante(labels == LABEL_AORTE, vg)
    sources["monai"] = (monai, mesurer(monai, espacement))
    affine, seuils = masque_densite(labels, ct, espacement, monai)
    densite = aorte_ascendante(affine, vg)
    sources["densite"] = (densite, mesurer(densite, espacement))
    details["densite"] = seuils
    ts = masque_totalsegmentator(reference)
    if ts is not None:
        aorte_ts, coeur_ts = ts
        # Le coeur de TotalSegmentator v2 est un seul label : le ventricule
        # gauche MONAI (meme grille) sert de repere pour l'ascendante.
        ascendante_ts = aorte_ascendante(aorte_ts, vg if vg.any() else coeur_ts)
        sources["totalsegmentator"] = (
            ascendante_ts,
            mesurer(ascendante_ts, espacement),
        )

    sortie = {
        "convention": CONVENTION,
        "avertissement": AVERTISSEMENT,
        "methode": "sections perpendiculaires a l'axe, grille 0,2 mm, masque isotrope 0,6 mm",
        "sources": {
            nom: {**mesure, **({"densite": details[nom]} if nom in details else {})}
            for nom, (_, mesure) in sources.items()
        },
    }
    (RESULTATS / "orion_diametre_aorte.json").write_text(
        json.dumps(sortie, indent=2), encoding="utf-8"
    )
    image_de_controle(sources, ct, espacement, RESULTATS / "orion_diametre_aorte.png")
    for nom, (_, mesure) in sources.items():
        m = mesure["maximum_aire"] if mesure["profil"] else None
        print(
            f"{nom:17s} {len(mesure['profil']):3d} sections, "
            + (
                f"maximum (equivalent aire) {m['diametre_aire_mm']} mm a {m['abscisse_mm']} mm"
                if m
                else "aucune mesure"
            )
        )
    if "densite" in details:
        print("seuils densite :", details["densite"])


if __name__ == "__main__":
    main()
