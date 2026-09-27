import time

import cv2
import mediapipe as mp
import numpy as np
import pygame.midi

# ---------- MUSIQUE ----------

pygame.midi.init()

id_midi = pygame.midi.get_default_output_id()

if id_midi == -1:
    raise RuntimeError("Aucun instrument MIDI n'a ete trouve.")

midi = pygame.midi.Output(id_midi)

# Piano acoustique : canal 0
midi.set_instrument(0, 0)

# Violon : canal 1
midi.set_instrument(40, 1)

accords = {
    "Do": [60, 64, 67],
    "Re": [62, 66, 69],
    "Fa": [65, 69, 72],
    "Sol": [67, 71, 74],
    "La mineur": [69, 72, 76],
}

sons = {
    # Main droite : piano
    ("Right", 1): "La mineur",
    ("Right", 2): "Sol",
    ("Right", 3): "Fa",
    ("Right", 4): "Do",
    ("Right", 5): "Re",

    # Main gauche : violon
    ("Left", 1): "La mineur",
    ("Left", 2): "Sol",
    ("Left", 3): "Fa",
}

notes_en_cours = []


def jouer_accord(nom_accord, canal, duree=1.5):
    notes = accords[nom_accord]

    for note in notes:
        midi.note_on(note, 85, canal)

    notes_en_cours.append({
        "notes": notes,
        "canal": canal,
        "fin": time.time() + duree
    })


def arreter_notes_finies():
    maintenant = time.time()

    for accord in notes_en_cours[:]:
        if maintenant >= accord["fin"]:
            for note in accord["notes"]:
                midi.note_off(note, 0, accord["canal"])

            notes_en_cours.remove(accord)


# ---------- CAMERA ----------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Impossible d'ouvrir la webcam. Ferme les autres applications qui l'utilisent."
    )

cv2.namedWindow("IMOMUSIC - Piano et violon", cv2.WINDOW_NORMAL)
cv2.resizeWindow("IMOMUSIC - Piano et violon", 1280, 720)


# ---------- RECONNAISSANCE DES MAINS ----------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def compter_doigts(main, cote):
    nombre = 0

    # Pouce
    if cote == "Right":
        if main.landmark[4].x < main.landmark[3].x:
            nombre += 1
    else:
        if main.landmark[4].x > main.landmark[3].x:
            nombre += 1

    # Index, majeur, annulaire et auriculaire
    bouts = [8, 12, 16, 20]
    articulations = [6, 10, 14, 18]

    for bout, articulation in zip(bouts, articulations):
        if main.landmark[bout].y < main.landmark[articulation].y - 0.02:
            nombre += 1

    return nombre


geste_candidat = {"Right": None, "Left": None}
debut_geste = {"Right": 0, "Left": 0}
dernier_geste_joue = {"Right": None, "Left": None}

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        try:
            success, image = camera.read()
        except cv2.error:
            success = False

        # Ne ferme pas l'application si la webcam rate une image.
        if not success:
            image = np.zeros((480, 640, 3), dtype=np.uint8)

            cv2.putText(
                image,
                "Attente de la camera...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.imshow("IMOMUSIC - Piano et violon", image)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        textes = {
            "Right": "Piano : main droite",
            "Left": "Violon : main gauche"
        }

        mains_detectees = set()
        maintenant = time.time()

        if result.multi_hand_landmarks and result.multi_handedness:
            for i, hand in enumerate(result.multi_hand_landmarks):
                cote = result.multi_handedness[i].classification[0].label
                mains_detectees.add(cote)

                mp_draw.draw_landmarks(
                    image,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                nombre_doigts = compter_doigts(hand, cote)
                geste = (cote, nombre_doigts)
                accord = sons.get(geste)

                instrument = "Piano" if cote == "Right" else "Violon"

                if accord:
                    textes[cote] = f"{instrument} : {accord}"

                    if geste != geste_candidat[cote]:
                        geste_candidat[cote] = geste
                        debut_geste[cote] = maintenant

                    if (
                        maintenant - debut_geste[cote] >= 0.4
                        and geste != dernier_geste_joue[cote]
                    ):
                        canal = 0 if cote == "Right" else 1
                        jouer_accord(accord, canal)

                        dernier_geste_joue[cote] = geste
                        textes[cote] = f"JOUE : {instrument} - {accord}"

                else:
                    textes[cote] = f"{instrument} : geste non attribue"
                    geste_candidat[cote] = None
                    dernier_geste_joue[cote] = None

        for cote in ["Right", "Left"]:
            if cote not in mains_detectees:
                geste_candidat[cote] = None
                dernier_geste_joue[cote] = None

        arreter_notes_finies()

        cv2.putText(
            image,
            textes["Right"],
            (40, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )

        cv2.putText(
            image,
            textes["Left"],
            (40, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 180, 100),
            3
        )

        cv2.imshow("IMOMUSIC - Piano et violon", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()

for accord in notes_en_cours:
    for note in accord["notes"]:
        midi.note_off(note, 0, accord["canal"])

midi.close()
pygame.midi.quit()