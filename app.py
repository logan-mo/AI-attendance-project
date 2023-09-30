import variables
from variables import (
    employee_names,
    trainer_path,
    workbook_path,
    detector_frontalface_default_path,
    dataset_path,
)
from datetime import datetime
import streamlit as st
from PIL import Image
import numpy as np
import xlsxwriter
import cv2
import os


def add_data(face_id):
    face_id = int(face_id)
    frame_placeholder = st.empty()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # frontal_face_default
    face_detector = cv2.CascadeClassifier(detector_frontalface_default_path)

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(
                "dataset/User." + str(face_id) + "." + str(count) + ".jpg",
                gray[y : y + h, x : x + w],
            )

            frame_placeholder.image(img, channels="BGR")

        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def train():
    # Path for face image database
    # path = 'dataset'
    path = dataset_path
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # detector_frontalface_default
    detector = cv2.CascadeClassifier(detector_frontalface_default_path)

    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            print(imagePath)

            PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale

            img_numpy = np.array(PIL_img, "uint8")

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            print("id", id)
            faces = detector.detectMultiScale(
                img_numpy,
                scaleFactor=1.3,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for x, y, w, h in faces:
                faceSamples.append(img_numpy[y : y + h, x : x + w])
                ids.append(id)

        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    # trainer
    recognizer.write(trainer_path)  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def take_attendance():
    frame_placeholder = st.empty()

    # to get unique values from attendnace list
    def unique(list1):
        x = np.array(list1)

    now = datetime.now()
    current_date_time = now.strftime("%d-%m-%Y %H-%M-%S")

    # Create a workbook and add a worksheet for each session of attendance.
    # workbook_path
    workbook_path = f"""{variables.workbook_path} {str(current_date_time)}.xlsx"""

    workbook = xlsxwriter.Workbook(workbook_path)

    worksheet = workbook.add_worksheet()
    worksheet.write("A1", "Serial No.")
    worksheet.write("B1", "Name")
    worksheet.write("C1", "Status")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # trainer_path
    recognizer.read(trainer_path)
    # detector_frontalface_default_path
    cascadePath = detector_frontalface_default_path

    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    print("Names:", employee_names)
    names = employee_names

    present_employees = []
    absent_employees = []

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

            # Check if confidence is less than 100 ==> "0" is perfect match
            if confidence < 100:
                # IF id has a name, label the image as that name, else just put out the index number
                try:
                    id = names[id]
                except IndexError as e:
                    id = id
                new_confidence = "  {0}%".format(round(100 - confidence))

                # writing attendance to excel sheet
                # only consider those students present whose confidence is more than 50%
                attendance_confidence = "  {0}".format(round(100 - confidence))
                if int(attendance_confidence) > 50:
                    # writing present students list
                    present_employees.append(id)

            elif confidence < 50:
                id = "unknown"
                new_confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            # print("Name of Student : " + str(id))
            # print("Confidence : " + str(confidence))
            cv2.putText(
                img, str(new_confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1
            )

        frame_placeholder.image(img, channels="BGR")

        k = cv2.waitKey(10) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break

    present_employees = np.unique(present_employees)

    # differnce of present and absent students list from original dataset
    # z=x xor y ; elements of x minus y
    absent_employees = set(names) ^ set(present_employees)
    absent_employees = list(absent_employees)
    print("\nPresent Students : ", present_employees)
    print("\nAbsent Students : ", absent_employees)

    # writing attenance to excel sheet
    row = 1
    col = 0
    count = 0
    for i in range(len(present_employees)):
        # serial no.
        worksheet.write(row, col, count + 1)
        # names
        worksheet.write(row, col + 1, present_employees[i])
        # status
        worksheet.write(row, col + 2, "P")
        row += 1
        count += 1

    for i in range(len(absent_employees)):
        worksheet.write(row, col, count + 1)
        worksheet.write(row, col + 1, absent_employees[i])
        worksheet.write(row, col + 2, "A")
        row += 1
        count += 1

    workbook.close()
    print("\n [INFO] Attendance Taken and Saved in Excel Sheet :) ")
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Attendance",
        page_icon="icon.png",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    local_css("streamlit_styling.css")

    st.title("Automated Attendance using AI")
    st.sidebar.header("Menu")

    face_id = st.sidebar.text_input("Enter new Employee id:")

    employee_Data = st.sidebar.button(
        "Add Employee Data", on_click=add_data, args=(face_id,)
    )
    checkbox_state = st.sidebar.button(
        "Train",
        on_click=train,
    )
    btn_report_gen = st.sidebar.button(
        "Take Attendance  ",
        on_click=take_attendance,
    )


main()
