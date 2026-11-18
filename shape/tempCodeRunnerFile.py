        display_width = 700
        display_height = 850

        display_image = cv2.resize(image, (display_width, display_height))

        cv2.imshow("MediaPipe Facial Landmarks", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
