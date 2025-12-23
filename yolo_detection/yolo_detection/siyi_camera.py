import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')
        
        # 압축 이미지 토픽 발행 (CompressedImage)
        # 보내주신 C++ 코드와 동일하게 '/image_raw/compressed' 사용
        self.pub_compressed = self.create_publisher(CompressedImage, '/image_raw/compressed', 10)
        
        self.bridge = CvBridge()
        
        # --- USB 카메라 설정 ---
        self.device_id = 0  # /dev/video0
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # 카메라 열기
        self.cap = cv2.VideoCapture(self.device_id)
        
        # 해상도 및 FPS 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        if not self.cap.isOpened():
            self.get_logger().error(f"카메라 {self.device_id}번을 열 수 없습니다.")
            # 혹시 모르니 1번 시도
            self.cap = cv2.VideoCapture(1)
            if self.cap.isOpened():
                 self.get_logger().info("1번 카메라 연결 성공!")
        else:
            self.get_logger().info(f"USB 카메라 연결 성공 ({self.width}x{self.height})")

        # 타이머 설정 (FPS에 맞춰 실행)
        self.timer = self.create_timer(1.0/self.fps, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # 이미지 압축 (JPEG Quality 80)
            self.publish_compressed(frame)
        else:
            self.get_logger().warn("프레임 수신 실패")
            
    def publish_compressed(self, frame):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.format = "jpeg"
        
        # JPEG 압축 (C++ 코드의 imencode 부분)
        # quality=80 설정
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_img = cv2.imencode('.jpg', frame, encode_param)
        
        if success:
            msg.data = np.array(encoded_img).tobytes()
            self.pub_compressed.publish(msg)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = UsbCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()