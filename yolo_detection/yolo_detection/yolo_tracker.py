import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import cv2
import socket
import struct
import numpy as np

from ultralytics import YOLO


class SiyiGimbalInterface:
    """
    UDP gimbal speed control.
    Packet format:
      0x55 0x66 CTRL LEN(2) SEQ(2) CMD(1) DATA(n) CRC16(2)

    CRC16: CRC-CCITT (poly=0x1021, init=0x0000), calculated over header+data (no CRC bytes).
    NOTE:
      If your C reference uses different init (e.g., 0xFFFF) or LEN definition,
      you must match it exactly or the gimbal will ignore packets.
    """

    STX0 = 0x55
    STX1 = 0x66
    CTRL = 0x01
    CMD_GIMBAL_SPEED = 0x07

    def __init__(self, ip: str = "192.168.144.25", port: int = 37260):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0  # uint16

    @staticmethod
    def _crc16_ccitt(data: bytes) -> int:
        crc = 0x0000
        poly = 0x1021
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc & 0xFFFF

    def _build_packet(self, cmd: int, payload: bytes) -> bytes:
        length = len(payload) & 0xFFFF
        seq = self.seq & 0xFFFF

        # Header: STX0, STX1, CTRL, LEN(u16), SEQ(u16), CMD(u8)
        header = struct.pack("<BBBHHB", self.STX0, self.STX1, self.CTRL, length, seq, cmd)
        body = header + payload

        crc = self._crc16_ccitt(body)
        packet = body + struct.pack("<H", crc)

        self.seq = (self.seq + 1) & 0xFFFF
        return packet

    def send_speed(self, yaw: int, pitch: int) -> None:
        # Clamp to typical SIYI range (signed int8)
        yaw = max(-100, min(100, int(yaw)))
        pitch = max(-100, min(100, int(pitch)))

        payload = struct.pack("<bb", yaw, pitch)  # 2 bytes signed
        packet = self._build_packet(self.CMD_GIMBAL_SPEED, payload)
        self.sock.sendto(packet, self.addr)


class YoloTrackerNode(Node):
    def __init__(self):
        super().__init__("yolo_tracker")

        # --- Parameters (easy tuning without editing code) ---
        self.declare_parameter("weights", "best.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("kp", 0.8)
        self.declare_parameter("deadzone", 0.05)
        self.declare_parameter("jpeg_quality", 60)
        self.declare_parameter("person_only", False)

        weights = self.get_parameter("weights").value
        image_topic = self.get_parameter("image_topic").value

        self.model = YOLO(weights)
        self.gimbal = SiyiGimbalInterface()
        self.bridge = CvBridge()

        self.kp = float(self.get_parameter("kp").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.person_only = bool(self.get_parameter("person_only").value)

        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.pub_result = self.create_publisher(CompressedImage, "/yolo/result/compressed", 10)

        self.get_logger().info(f"YOLO Tracker Started. weights={weights}, topic={image_topic}")

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        results = self.model.predict(frame, conf=self.conf, verbose=False, imgsz=self.imgsz)

        target = None
        min_dist = float("inf")

        for r in results:
            for box in r.boxes:
                if self.person_only:
                    # YOLO class 0 is often "person" in COCO, but your custom model may differ.
                    if int(box.cls[0]) != 0:
                        continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                dist = (cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2

                if dist < min_dist:
                    min_dist = dist
                    target = (x1, y1, x2, y2)

        if target is not None:
            x1, y1, x2, y2 = target
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Normalized error in range roughly [-0.5, 0.5]
            err_x = (cx / float(w)) - 0.5
            err_y = (cy / float(h)) - 0.5

            # Deadzone
            if abs(err_x) < self.deadzone:
                err_x = 0.0
            if abs(err_y) < self.deadzone:
                err_y = 0.0

            # Speed command (signs may need flipping depending on gimbal axis convention)
            yaw_cmd = int(-err_x * 100.0 * self.kp)
            pitch_cmd = int(-err_y * 100.0 * self.kp)

            self.gimbal.send_speed(yaw_cmd, pitch_cmd)

            # Visualization
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        else:
            self.gimbal.send_speed(0, 0)

        self.publish_compressed(frame)

    def publish_compressed(self, frame):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, enc = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.asarray(enc).tobytes()
        self.pub_result.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
