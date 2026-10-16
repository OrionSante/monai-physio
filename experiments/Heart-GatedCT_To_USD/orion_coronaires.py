"""Orion : coronaires extraites du scanner injecte (sans modele sous licence).

A lancer APRES orion_cavites_glb.py (il lit sa segmentation), depuis
l'environnement monai-physio :

    python orion_coronaires.py

TotalSegmentator ``coronary_arteries`` est sous licence ; ici, methode classique
et tracable, sur l'image de reference (results/slice_fixed.mha) :

1. Recadrage autour du coeur (myocarde + 4 cavites + racine aortique).
2. Filtre de vaisseaux de Frangi (structures tubulaires claires, rayons ~0,6 a
   2 mm) sur l'image en HU.
3. Candidats : tubulaires, rehausses (lumiere injectee), hors des structures
   deja segmentees (cavites, gros vaisseaux, veines...), a moins de quelques mm
   de la surface du coeur.
4. Croissance a partir de la racine aortique : on ne garde que ce qui est
   connecte a l'aorte pres du coeur (ostia coronaires). Les veines cardiaques,
   peu rehaussees a ce temps arteriel, et les veines pulmonaires (reliees a
   l'oreillette gauche) ne sont pas retenues.
5. Tri par la forme : seuls les tubes fins et longs sont gardes (rayon maximal
   <= RAYON_RETENU_MM, squelette >= LONGUEUR_MIN_MM). Regle fixee apres controle
   visuel de ce scanner : sur les trois composantes trouvees, une seule (tube
   d'~11 cm, rayon <= 1,3 mm) ressemblait a une artere coronaire ; une nappe
   contre la racine aortique (rayon 2,1 mm) et un enchevetrement du sillon
   posterieur (1,8 mm) sont ecartes. Le reseau coronaire est donc INCOMPLET.

Sortie : results/orion_coronaires.labelmap.mha (etiquette 50) et un apercu
results/orion_coronaires.apercu.png (projection d'intensite maximale) pour le
controle visuel. Le resultat est une extraction automatique, non validee par un
radiologue.
"""
from __future__ import annotations

from pathlib import Path

import itk
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.morphology import skeletonize

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "results"
ETIQUETTE = 50

AORTE, MYOCARDE = 7, 44
COEUR = (44, 45, 46, 47, 48)
MARGE_MM = 15.0
HU_LUMIERE = 250.0  # lumiere injectee (aorte ~650 HU, myocarde ~140 HU)
SIGMAS_VOXELS = (1.0, 1.5, 2.0, 3.0)
DISTANCE_COEUR_MM = 8.0  # les coronaires courent dans les sillons, a la surface
RACINE_MM = 30.0  # partie de l'aorte retenue comme point de depart (ostia)
VOLUME_MIN_MM3 = 150.0
PERCENTILE_TUBULAIRE = 70
RAYON_MAX_MM = 2.5
RAYON_RETENU_MM = 1.5
LONGUEUR_MIN_MM = 50.0


def charger():
    image = itk.imread(str(RESULTATS / "slice_fixed.mha"), itk.F)
    cavites = itk.array_from_image(itk.imread(str(RESULTATS / "orion_cavites.labelmap.mha")))
    # Toutes les structures du bundle (104), sur la grille de reference : sert
    # a exclure ce qui est deja nomme (veines caves, veines pulmonaires...).
    brut = next((RESULTATS / "orion_wholebody").rglob("*_trans.nii.gz"))
    seg = itk.imread(str(brut), itk.F)
    seg = itk.resample_image_filter(
        seg, reference_image=image, use_reference_image=True,
        interpolator=itk.NearestNeighborInterpolateImageFunction.New(seg),
    )
    return image, cavites, itk.array_from_image(seg).astype(np.int16)


def extraire(image, cavites, toutes):
    hu = itk.array_from_image(image)  # z, y, x
    esp = np.asarray(itk.spacing(image))[::-1]  # z, y, x
    voxel_mm3 = float(np.prod(esp))

    coeur = np.isin(cavites, COEUR)
    zz, yy, xx = np.nonzero(coeur | (cavites == AORTE) & (ndi.distance_transform_edt(~coeur, sampling=esp) < RACINE_MM))
    marge = np.ceil(MARGE_MM / esp).astype(int)
    bas = np.maximum([zz.min(), yy.min(), xx.min()] - marge, 0)
    haut = np.minimum([zz.max(), yy.max(), xx.max()] + marge + 1, hu.shape)
    sl = tuple(slice(int(b), int(h)) for b, h in zip(bas, haut))
    hu_c, cav_c, tout_c = hu[sl], cavites[sl], toutes[sl]
    print(f"[coronaires] recadrage {hu_c.shape} voxels", flush=True)

    coeur_c = np.isin(cav_c, COEUR)
    dist_coeur = ndi.distance_transform_edt(~coeur_c, sampling=esp)
    aorte_c = cav_c == AORTE
    racine = aorte_c & (dist_coeur < RACINE_MM)

    # Frangi sur l'image fenetree : les os et le calcium saturent a 1.
    norm = np.clip((hu_c + 200.0) / 1100.0, 0.0, 1.0).astype(np.float32)
    print("[coronaires] filtre de Frangi...", flush=True)
    cache = RESULTATS / "orion_coronaires.frangi.npy"
    if cache.exists() and np.load(cache, mmap_mode="r").shape == norm.shape:
        vas = np.load(cache)
    else:
        vas = frangi(norm, sigmas=SIGMAS_VOXELS, black_ridges=False).astype(np.float32)
        np.save(cache, vas)
    # Exclusion : tout ce qui est deja nomme, sauf le myocarde (les coronaires
    # en touchent le bord), un peu dilate pour ne pas lecher les parois.
    nomme = ~np.isin(tout_c, [0, MYOCARDE])
    exclu = ndi.binary_dilation(nomme, iterations=1)
    # Les coronaires courent a la surface externe du coeur : le sang des
    # cavites que la segmentation (1,5 mm) a manque est a l'interieur du
    # volume plein du coeur, ferme puis rempli.
    plein = ndi.binary_fill_holes(ndi.binary_closing(coeur_c, structure=np.ones((3, 3, 3), bool), iterations=3))
    zone = (hu_c > HU_LUMIERE) & ~exclu & ~plein & (dist_coeur < DISTANCE_COEUR_MM)
    # Seuil relatif : la moitie la moins tubulaire des voxels rehausses de la
    # zone (bords de cavites mal segmentes, graisse partielle) est ecartee.
    seuil = float(np.percentile(vas[zone], PERCENTILE_TUBULAIRE))
    print(f"[coronaires] vesselness : seuil {seuil:.4f} (p{PERCENTILE_TUBULAIRE} de {int(zone.sum())} voxels)", flush=True)
    candidat = zone & (vas > seuil)
    # Trop epais pour une coronaire (rayon > RAYON_MAX_MM) : poche de sang,
    # auricule, veine... on retire ces parties et leur voisinage.
    epais = ndi.distance_transform_edt(candidat, sampling=esp) > RAYON_MAX_MM
    candidat &= ~ndi.binary_dilation(epais, iterations=int(np.ceil(RAYON_MAX_MM / esp.min())) + 1)

    # Depart : candidats au contact de la racine aortique.
    contact = ndi.binary_dilation(racine, iterations=3)
    etiq, n = ndi.label(candidat, structure=np.ones((3, 3, 3), bool))
    graines = np.unique(etiq[contact & candidat])
    graines = graines[graines > 0]
    tailles = ndi.sum(np.ones_like(etiq), etiq, graines) * voxel_mm3
    gardes = [int(g) for g, t in zip(graines, tailles) if t >= VOLUME_MIN_MM3]
    print(f"[coronaires] {n} composantes candidates, {len(graines)} au contact de l'aorte, "
          f"{len(gardes)} retenues ({', '.join(f'{t:.0f}' for t in sorted(tailles, reverse=True)[:6])} mm3)", flush=True)
    masque_c = np.isin(etiq, gardes)
    # Fermeture legere : recolle les segments coupes par un voxel bruite.
    ferme = ndi.binary_closing(masque_c, structure=np.ones((3, 3, 3), bool))
    masque_c |= ferme & (hu_c > HU_LUMIERE * 0.8) & ~exclu
    masque_c = trier_par_forme(masque_c, esp)
    sortie = np.zeros(hu.shape, np.uint8)
    sortie[sl][masque_c] = ETIQUETTE
    print(f"[coronaires] {int(masque_c.sum())} voxels ({masque_c.sum() * voxel_mm3 / 1000:.2f} mL)", flush=True)
    return sortie, sl, hu_c, masque_c, cav_c


def trier_par_forme(masque: np.ndarray, esp: np.ndarray) -> np.ndarray:
    """Ne garde que les composantes fines et longues (voir etape 5)."""
    etiq, n = ndi.label(masque, structure=np.ones((3, 3, 3), bool))
    garde = np.zeros_like(masque)
    for i, sl in enumerate(ndi.find_objects(etiq), start=1):
        m = etiq[sl] == i
        longueur = float(skeletonize(m).sum() * esp.mean())
        rayon = float(ndi.distance_transform_edt(m, sampling=esp).max())
        ok = rayon <= RAYON_RETENU_MM and longueur >= LONGUEUR_MIN_MM
        print(f"[coronaires] composante {i} : squelette {longueur:.0f} mm, rayon max {rayon:.1f} mm"
              f" -> {'retenue' if ok else 'ecartee'}", flush=True)
        if ok:
            garde[sl] |= m
    return garde


def apercu(hu_c, masque_c, cav_c, chemin: Path):
    """Projection d'intensite maximale (vue de face) : scanner en gris, coronaires en rouge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, axe, titre in ((axes[0], 1, "face (MIP antero-posterieure)"), (axes[1], 2, "profil (MIP laterale)")):
        mip = np.clip(hu_c, -200, 900).max(axis=axe)
        ax.imshow(mip, cmap="gray", origin="lower", aspect="auto")
        cor = masque_c.max(axis=axe)
        ax.imshow(np.ma.masked_where(~cor, cor), cmap="autumn", alpha=0.9, origin="lower", aspect="auto")
        ax.set_title(titre)
        ax.axis("off")
    fig.suptitle("Coronaires extraites (rouge) - controle visuel")
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    print(f"[coronaires] apercu {chemin}")


if __name__ == "__main__":
    image, cavites, toutes = charger()
    sortie, sl, hu_c, masque_c, cav_c = extraire(image, cavites, toutes)
    lm = itk.image_from_array(sortie)
    lm.CopyInformation(image)
    itk.imwrite(lm, str(RESULTATS / "orion_coronaires.labelmap.mha"), compression=True)
    apercu(hu_c, masque_c, cav_c, RESULTATS / "orion_coronaires.apercu.png")
