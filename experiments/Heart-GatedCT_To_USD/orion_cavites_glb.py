"""Orion : cavites du coeur (wholeBody_ct_segmentation) -> .glb 4D pour le web.

A lancer APRES 1-register_images.py (il lit ses transformations), depuis
l'environnement monai-physio :

    python orion_cavites_glb.py

1. Segmente l'image de reference (results/slice_fixed.mha) avec le bundle MONAI
   ``wholeBody_ct_segmentation`` (Apache 2.0, SegResNet 1,5 mm, 104 structures
   TotalSegmentator v1) : myocarde, 4 cavites, aorte, artere pulmonaire.
2. Extrait les contours de ces structures (ProcessContours, comme le tutoriel).
3. Les deforme pour chaque phase avec les transformations du tutoriel
   (slice_XXX.reg_dynamic_anatomy.forward.hdf) : meme topologie a chaque phase.
4. Ecrit un .glb : un maillage nomme par structure, la reference comme forme de
   base, une morph target par phase et une animation en boucle d'un cycle (1 s,
   a mettre a l'echelle de la frequence cardiaque cote web).

Le modele est celui d'un patient du jeu de donnees public Slicer-Heart-CT, pas
d'un patient Orion. Coronaires : aucun modele libre ne les segmente ; si
orion_coronaires.py a ete lance, son unique vaisseau retenu (extraction
classique, reseau incomplet, non valide) est ajoute comme structure
"Coronary".
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import itk
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from monai_physio.process_contours import ProcessContours

ICI = Path(__file__).resolve().parent
RESULTATS = ICI / "results"
BUNDLE = Path(r"E:\work\monai-bundles\model-zoo\models\wholeBody_ct_segmentation")
SORTIE_GLB = Path(r"E:\work\orion-team-export\frontend\public\models\heart_cardiotwin_4d.glb")

# Doit suivre N_FRAMES / FRAME_STEP de 1-register_images.py.
N_FRAMES, FRAME_STEP = 21, 3
PHASES = list(range(0, N_FRAMES, FRAME_STEP))

# Etiquettes wholeBody_ct_segmentation (metadata.json) -> nom du sous-maillage.
STRUCTURES = {
    7: "Aorta",
    44: "Myocardium",
    45: "LeftAtrium",
    46: "LeftVentricle",
    47: "RightAtrium",
    48: "RightVentricle",
    49: "PulmonaryArtery",
    50: "Coronary",  # orion_coronaires.py, pas le bundle
}
CORONAIRES = RESULTATS / "orion_coronaires.labelmap.mha"
COULEURS = {  # base PBR neutre ; le front applique ses propres materiaux
    "Aorta": (0.60, 0.63, 0.65), "Myocardium": (0.55, 0.12, 0.15),
    "LeftAtrium": (0.62, 0.20, 0.22), "LeftVentricle": (0.58, 0.14, 0.17),
    "RightAtrium": (0.45, 0.20, 0.30), "RightVentricle": (0.48, 0.16, 0.24),
    "PulmonaryArtery": (0.45, 0.52, 0.65), "Coronary": (0.70, 0.08, 0.12),
}
SOMMETS_CIBLES = 6000  # par structure, apres simplification
LISSAGE_ITERATIONS = 60
LISSAGE_BANDE = 0.02  # plus petit = plus lisse

# Recadrage autour du coeur avant inference : 105 cartes de probabilite sur le
# volume entier depassent la memoire d'une machine a 16 Go. Etiquettes
# TotalSegmentator du masque de reference (51 coeur, 61 auricule gauche).
ETIQUETTES_COEUR_TS = (51, 61)
MARGE_MM = 30.0


def recadrer_sur_coeur() -> Path:
    """Sous-volume de slice_fixed.mha autour du coeur (masque de l'etape 1)."""
    sortie = RESULTATS / "orion_coeur_recadre.mha"
    if sortie.exists():
        return sortie
    image = itk.imread(str(RESULTATS / "slice_fixed.mha"))
    masque = itk.array_from_image(itk.imread(str(RESULTATS / "slice_fixed_mask.mha")))
    zz, yy, xx = np.nonzero(np.isin(masque, ETIQUETTES_COEUR_TS))
    if not len(zz):
        raise RuntimeError("coeur absent du masque de reference : recadrage impossible")
    esp = np.asarray(itk.spacing(image))  # x, y, z
    taille = np.asarray(itk.size(image))
    marge = np.ceil(MARGE_MM / esp).astype(int)
    bas = np.maximum([xx.min(), yy.min(), zz.min()] - marge, 0)
    haut = np.minimum([xx.max(), yy.max(), zz.max()] + marge + 1, taille)
    region = itk.ImageRegion[3]()
    region.SetIndex([int(v) for v in bas])
    region.SetSize([int(v) for v in haut - bas])
    recadre = itk.region_of_interest_image_filter(image, region_of_interest=region)
    itk.imwrite(recadre, str(sortie), compression=True)
    print(f"[cavites] recadrage {tuple(int(v) for v in haut - bas)} voxels sur {tuple(int(v) for v in taille)}")
    return sortie


def segmenter_cavites() -> Path:
    """Inference du bundle sur l'image de reference (CPU, sans AMP)."""
    sortie = RESULTATS / "orion_wholebody"
    deja = list(sortie.rglob("*_trans.nii.gz"))
    if deja:
        print(f"[cavites] segmentation deja faite : {deja[0].name}")
        return deja[0]
    entree = recadrer_sur_coeur().as_posix()
    cmd = [
        sys.executable, "-m", "monai.bundle", "run",
        "--config_file", str(BUNDLE / "configs" / "inference.json"),
        "--bundle_root", str(BUNDLE),
        "--datalist", f"['{entree}']",
        "--output_dir", sortie.as_posix(),
        "--dataloader#num_workers", "0",
        "--evaluator#amp", "False",
        # Poids enregistres sur GPU : machine sans CUDA.
        "--checkpointloader#map_location", "cpu",
    ]
    print("[cavites] inference wholeBody_ct_segmentation (CPU)...", flush=True)
    subprocess.run(cmd, check=True)
    return next(sortie.rglob("*_trans.nii.gz"))


def labelmap_cardiaque(chemin: Path, reference: itk.Image) -> itk.Image:
    """Ne garde que les structures retenues, sur la grille de l'image de reference."""
    seg = itk.imread(str(chemin), itk.F)
    seg = itk.resample_image_filter(
        seg, reference_image=reference, use_reference_image=True,
        interpolator=itk.NearestNeighborInterpolateImageFunction.New(seg),
    )
    arr = itk.array_from_image(seg).astype(np.int16)
    # L'etiquette 50 du bundle est une autre structure : seule celle
    # d'orion_coronaires.py vaut "Coronary".
    garde = np.where(np.isin(arr, [k for k in STRUCTURES if k != 50]), arr, 0).astype(np.uint8)
    if CORONAIRES.exists():
        cor = itk.array_from_image(itk.imread(str(CORONAIRES))) > 0
        garde[cor & (garde == 0)] = 50
    else:
        print("[cavites] pas de coronaires (orion_coronaires.py non lance)")
    for lab, nom in STRUCTURES.items():
        print(f"[cavites] {nom:16s} {int((garde == lab).sum()):>9d} voxels")
    image = itk.image_from_array(garde)
    image.CopyInformation(reference)
    return image


def contours_par_phase(labelmap: itk.Image) -> tuple[pv.PolyData, dict[int, pv.PolyData]]:
    con = ProcessContours()
    base = con.extract_contours(labelmap)
    base.save(RESULTATS / "orion_cavites.reference.vtp")
    phases = {}
    for i in PHASES:
        tfm = RESULTATS / f"slice_{i:03d}.reg_dynamic_anatomy.forward.hdf"
        t = itk.transformread(str(tfm))[0]
        p = con.transform_contours(base, t)
        if p.n_points != base.n_points:
            raise RuntimeError(f"phase {i} : topologie differente de la reference")
        p.save(RESULTATS / f"slice_{i:03d}.orion_cavites.vtp")
        phases[i] = p
        print(f"[cavites] phase {i:02d} deformee", flush=True)
    return base, phases


def vers_gltf(p: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """LPS (mm) -> glTF (m) : X = gauche du patient, Y = haut, Z = avant."""
    q = (p - centre) * 0.001
    return np.stack([q[:, 0], q[:, 2], -q[:, 1]], axis=1).astype(np.float32)


def sous_maillages(base: pv.PolyData, phases: dict[int, pv.PolyData]):
    """Pour chaque structure : maillage simplifie et deplacement de chaque phase."""
    labels = np.asarray(base.cell_data["boundary_labels"]).reshape(base.n_cells, -1)
    pts_base = np.asarray(base.points)
    arbre = cKDTree(pts_base)
    for lab, nom in STRUCTURES.items():
        faces = np.where((labels == lab).any(axis=1))[0]
        if not len(faces):
            print(f"[glb] {nom} : absent de la segmentation, ignore")
            continue
        morceau = base.extract_cells(faces).extract_surface().triangulate().clean()
        # Lissage de Taubin : efface les marches de voxels sans retrecir le volume
        # (un lissage laplacien simple amincirait les parois et l'aorte).
        morceau = morceau.smooth_taubin(n_iter=LISSAGE_ITERATIONS, pass_band=LISSAGE_BANDE)
        ratio = max(0.0, 1.0 - SOMMETS_CIBLES / max(morceau.n_points, 1))
        simple = morceau.decimate(ratio) if ratio > 0 else morceau
        simple = simple.compute_normals(auto_orient_normals=True, consistent_normals=True)
        # Chaque sommet simplifie suit le sommet d'origine le plus proche.
        _, idx = arbre.query(np.asarray(simple.points))
        deplacements = {i: np.asarray(ph.points)[idx] - pts_base[idx] for i, ph in phases.items()}
        tri = np.asarray(simple.faces).reshape(-1, 4)[:, 1:].astype(np.uint32)
        yield nom, np.asarray(simple.points), np.asarray(simple.point_data["Normals"]), tri, deplacements


def ecrire_glb(parties, centre: np.ndarray, chemin: Path) -> None:
    binaire = bytearray()
    vues, accesseurs, maillages, noeuds, materiaux = [], [], [], [], []

    def ajouter(tableau: np.ndarray, cible: int | None, type_: str, comp: int, minmax=False) -> int:
        while len(binaire) % 4:
            binaire.append(0)
        donnees = np.ascontiguousarray(tableau).tobytes()
        vue = {"buffer": 0, "byteOffset": len(binaire), "byteLength": len(donnees)}
        if cible:
            vue["target"] = cible
        vues.append(vue)
        binaire.extend(donnees)
        acc = {"bufferView": len(vues) - 1, "componentType": comp,
               "count": int(tableau.shape[0]), "type": type_}
        if minmax:
            acc["min"] = [float(v) for v in tableau.min(axis=0)]
            acc["max"] = [float(v) for v in tableau.max(axis=0)]
        accesseurs.append(acc)
        return len(accesseurs) - 1

    FLOAT, UINT = 5126, 5125
    for nom, pts, normales, tri, depl in parties:
        pos = vers_gltf(pts, centre)
        nrm = np.stack([normales[:, 0], normales[:, 2], -normales[:, 1]], axis=1).astype(np.float32)
        cibles = []
        for i in PHASES:
            d = depl[i] * 0.001
            d = np.stack([d[:, 0], d[:, 2], -d[:, 1]], axis=1).astype(np.float32)
            cibles.append({"POSITION": ajouter(d, 34962, "VEC3", FLOAT, minmax=True)})
        materiaux.append({"name": nom, "pbrMetallicRoughness": {
            "baseColorFactor": [*COULEURS[nom], 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.35}})
        maillages.append({
            "name": nom,
            "primitives": [{
                "attributes": {"POSITION": ajouter(pos, 34962, "VEC3", FLOAT, minmax=True),
                               "NORMAL": ajouter(nrm, 34962, "VEC3", FLOAT)},
                "indices": ajouter(tri.reshape(-1), 34963, "SCALAR", UINT),
                "material": len(materiaux) - 1,
                "targets": cibles,
            }],
            "weights": [0.0] * len(PHASES),
            "extras": {"targetNames": [f"phase_{i:02d}" for i in PHASES]},
        })
        noeuds.append({"name": nom, "mesh": len(maillages) - 1})
        print(f"[glb] {nom:16s} {len(pos):>6d} sommets, {len(tri):>6d} triangles")

    # Animation : une phase apres l'autre sur un cycle d'1 s, retour a la premiere.
    n = len(PHASES)
    temps = np.linspace(0.0, 1.0, n + 1).astype(np.float32)
    poids = np.zeros((n + 1, n), dtype=np.float32)
    for k in range(n + 1):
        poids[k, k % n] = 1.0
    t_acc = ajouter(temps.reshape(-1, 1), None, "SCALAR", FLOAT, minmax=True)
    w_acc = ajouter(poids.reshape(-1, 1), None, "SCALAR", FLOAT)
    animation = {"name": "battement", "samplers": [], "channels": []}
    for k in range(len(noeuds)):
        animation["samplers"].append({"input": t_acc, "output": w_acc, "interpolation": "LINEAR"})
        animation["channels"].append({"sampler": k, "target": {"node": k, "path": "weights"}})

    racine = {"name": "CardioTwin", "children": list(range(len(noeuds)))}
    noeuds.append(racine)
    gltf = {
        "asset": {"version": "2.0", "generator": "Orion orion_cavites_glb.py (MONAI Physio + wholeBody_ct_segmentation)"},
        "scene": 0, "scenes": [{"nodes": [len(noeuds) - 1]}],
        "nodes": noeuds, "meshes": maillages, "materials": materiaux,
        "accessors": accesseurs, "bufferViews": vues,
        "buffers": [{"byteLength": len(binaire)}], "animations": [animation],
        "extras": {"phases": PHASES, "phase_reference": 7, "source": "Slicer-Heart-CT (jeu public)"},
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binaire.extend(b"\0" * (-len(binaire) % 4))
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binaire)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binaire), 0x004E4942) + binaire)
    print(f"[glb] ecrit {chemin} ({chemin.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    reference = itk.imread(str(RESULTATS / "slice_fixed.mha"), itk.F)
    labelmap = labelmap_cardiaque(segmenter_cavites(), reference)
    itk.imwrite(labelmap, str(RESULTATS / "orion_cavites.labelmap.mha"), compression=True)
    base, phases = contours_par_phase(labelmap)
    coeur = np.isin(np.asarray(base.cell_data["boundary_labels"]).reshape(base.n_cells, -1), [44, 45, 46, 47, 48]).any(axis=1)
    centre = np.asarray(base.extract_cells(np.where(coeur)[0]).points).mean(axis=0)
    ecrire_glb(list(sous_maillages(base, phases)), centre, SORTIE_GLB)
