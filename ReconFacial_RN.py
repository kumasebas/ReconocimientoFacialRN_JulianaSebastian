import face_recognition
import os
import cv2
import numpy

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = 'cnn'

video = cv2.VideoCapture('videoRN.mp4')

print('Procesando caras conocidas...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if name != '.DS_Store':
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
            if filename != '.DS_Store':
                image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(name)

metodoDeReconocimiento = input("¿Desea utilizar una imagen o un video? i = imagen, v = video\n")

if metodoDeReconocimiento == 'i':
    print('Procesando caras no reconocidas...')
    for filename in os.listdir(UNKNOWN_FACES_DIR):
        if filename != '.DS_Store':

            print(f'Filename {filename}', end='')
            image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
            locations = face_recognition.face_locations(image, model=MODEL)
            encodings = face_recognition.face_encodings(image, locations)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            print(f', found {len(encodings)} face(s)')
            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:
                    match = known_names[results.index(True)]
                    print(f' - {match} from {results}')
                    top_left = (face_location[3], face_location[0])
                    bottom_right = (face_location[1], face_location[2])
                    color = [0, 255, 0]
                    cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                    top_left = (face_location[3], face_location[2])
                    bottom_right = (face_location[1], face_location[2] + 22)
                    cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                    cv2.putText(image, match, (face_location[3] + 5, face_location[2] + 15), cv2.FONT_ITALIC, 0.5, (0, 0, 0), FONT_THICKNESS)
                    cv2.putText(image, f'{len(encodings)} cara(s)', (0, 15), cv2.FONT_ITALIC, 0.5, (0, 255, 0), FONT_THICKNESS)
        cv2.imshow(filename, image)
        cv2.waitKey(0)

if metodoDeReconocimiento == 'v':
    print('Reconociendo video...')
    while True:
        ret, image = video.read()

        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)

        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            match = None
            if True in results:
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
                color = [0, 255, 0]
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (face_location[3] + 5, face_location[2] + 15), cv2.FONT_ITALIC, 0.5, (0, 0, 0), FONT_THICKNESS)
                cv2.putText(image, f'{len(encodings)} cara(s)', (0, 15), cv2.FONT_ITALIC, 0.5, (0, 255, 0), FONT_THICKNESS)
        cv2.imshow(filename, image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break