from object_tracking_pipeline.detection_models.onnx_model.onnx_prediction import detect_objects_and_draw_boxes
import cv2

# model = YOLO('yolov8n.pt')

def main():
    # Инициализация камеры
    camera = cv2.VideoCapture(0)  # Используем камеру с индексом 0 (обычно встроенную в ноутбуки или внешнюю камеру)

    if not camera.isOpened():
        print("Камера не найдена или не может быть открыта.")
        return

    while True:
        # Чтение кадра с камеры
        ret, frame = camera.read()

        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        # Отображение кадра в окне
        # frame = model.track(frame, show=True)
        frame = detect_objects_and_draw_boxes(frame)
        cv2.imshow("Camera Feed", frame)

        # Для выхода из цикла, нажмите клавишу 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрытие камеры и окна
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
