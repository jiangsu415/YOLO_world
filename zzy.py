from ultralytics import YOLOWorld

if __name__ == '__main__':
    # Initialize a YOLO-World model
    model = YOLOWorld('/root/zzy/YOLO_world/yolov8s-world.pt')  # or choose yolov8m/l-world.pt

    # Define custom classes
    model.set_classes(["person","bus"])

    # Execute prediction for specified categories on an image
    results = model.predict('/root/zzy/YOLO_world/data/images/bus.jpg')

    # Show results
    print(results[0])