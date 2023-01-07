import cv2
import torch
import argparse
import os
from models.common import DetectMultiBackend


def objects_detect(mode, img):
    model_name = 'street_objects_detect.pt'
    model = torch.hub.load('.', 'custom', model_name, source='local')

    if mode == 0:
        capture = cv2.VideoCapture(0)

        while True:
            ret, frame = capture.read()

            results = model(frame, size=640)
            stride, names, pt = model.stride, model.names, model.pt

            if len(results.xyxy[0]):
                for object in results.xyxy[0].tolist():
                    if int(object[4] * 100) >= 30:
                        cv2.rectangle(frame, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])),
                                      (0, 255, 149), 2)
                        cv2.putText(frame, f'{names[int(object[5])]} - {int(object[4] * 100)}%',
                                    (int(object[0]) + 20, int(object[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 149),
                                    2)

                        print('--------------', f'Класс: {names[int(object[5])]}',
                              f'Координаты: {(int(object[0]), int(object[1]))} и {(int(object[2]), int(object[3]))}',
                              '--------------', sep='\n', end='\n\n')

            cv2.imshow('В режиме реального времени', frame)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break

    elif mode == 1 and img:
        image = cv2.imread(img)
        results = model(image, size=640)
        stride, names, pt = model.stride, model.names, model.pt

        if len(results.xyxy[0]):
            for object in results.xyxy[0].tolist():
                if int(object[4] * 100) >= 30:
                    cv2.rectangle(image, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])),
                                  (0, 255, 149), 2)
                    cv2.putText(image, f'{names[int(object[5])]} - {int(object[4] * 100)}%',
                                (int(object[0]) + 20, int(object[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 149),
                                2)

                    print('--------------', f'Класс: {names[int(object[5])]}',
                          f'Координаты: {(int(object[0]), int(object[1]))} и {(int(object[2]), int(object[3]))}',
                          '--------------', sep='\n', end='\n\n')

        cv2.imshow('Изображение', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='режим')
    parser.add_argument('--image', help='изображение')

    args = parser.parse_args()

    objects_detect(int(args.mode), args.image)