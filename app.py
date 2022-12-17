from builtins import print, open, enumerate, str, int, zip, dict

import cv2
from flask import Flask, Response, render_template, jsonify
import mediapipe as mp
from flask import request
import os
import json
import pandas as pd

app = Flask(__name__)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

up = False

muscle_push_0 = 'Chest Press'
muscle_push_1 = 'Shoulder Press'

muscle_pull_1 = 'Bicep Curls'
muscle_pull_2 = 'Deadlifts'

with open("dummyjson1.json") as f:
    data = json.load(f)
df = pd.DataFrame(data)
l1 = []
l2 = []
for i in df['username']:
    l1.append(i)

for i in df['password']:
    l2.append(i)
creds = dict(zip(l1, l2))
print(creds)


def deadLifts_fat(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[15][1] < points[23][1] and points[16][1] < points[24][1]:
                    up = True
                    jit += 1
                elif points[15][1] > points[23][1] and points[16][1] > points[24][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def deadLifts_muscle(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[15][1] < points[23][1] and points[16][1] < points[24][1]:
                    up = True
                    jit += 1
                elif points[15][1] > points[23][1] and points[16][1] > points[24][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def bicepCurls_fat(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[17][1] < points[11][1] and points[18][1] < points[12][1]:
                    up = True
                    jit += 1
                elif points[17][1] > points[11][1] and points[18][1] > points[12][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def bicepCurls_muscle(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[17][1] < points[11][1] and points[18][1] < points[12][1]:
                    up = True
                    jit += 1
                elif points[17][1] > points[11][1] and points[18][1] > points[12][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def shoulderPress_fat(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[14][1] < points[12][1] and points[13][1] < points[11][1]:
                    up = True
                    jit += 1
                elif points[14][1] > points[12][1] and points[13][1] > points[11][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def shoulderPress_muscle(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[14][1] < points[12][1] and points[13][1] < points[11][1]:
                    up = True
                    jit += 1
                elif points[14][1] > points[12][1] and points[13][1] > points[11][1]:
                    up = False

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def chestPress_fat(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (800, 670))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[16][1] < points[12][1] and points[15][1] < points[11][1]:
                    up = True
                    jit += 1
                elif points[16][1] > points[12][1] and points[15][1] > points[11][1]:
                    up = False

            print(jit)
            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


def chestPress_muscle(up, repCap):
    cam = cv2.VideoCapture(0)
    jit = 0
    while 1:
        success, img = cam.read()

        if success == True:
            img = cv2.resize(img, (750, 625))
            img = cv2.flip(img, 1)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                points = {}
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points[id] = (cx, cy)

                if not up and points[16][1] < points[12][1] and points[15][1] < points[11][1]:
                    up = True
                    jit += 1
                elif points[16][1] > points[12][1] and points[15][1] > points[11][1]:
                    up = False

            print(jit)

            if jit >= repCap:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
            else:
                cv2.putText(img, str(jit), (600, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            imgencode = cv2.imencode('.jpg', img)[1]
            frame = imgencode.tobytes()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            print('finished')
            break


# ------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/pullWorkout_2_fat')
def pullWorkout_2_fat():
    return Response(deadLifts_fat(up, 10), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pullWorkout_1_fat')
def pullWorkout_1_fat():
    return Response(bicepCurls_fat(up, 10), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fatPull_2')
def fatPull_2():
    return render_template('pulling2.html', workout=muscle_pull_2)


@app.route('/fatPull_1')
def fatPull_1():
    return render_template('pulling1.html', workout=muscle_pull_1)


@app.route('/pullWorkout_2_muscle')
def pullWorkout_2_muscle():
    return Response(deadLifts_muscle(up, 8), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pullWorkout_1_muscle')
def pullWorkout_1_muscle():
    return Response(bicepCurls_muscle(up, 8), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/musclePull_2')
def musclePull_2():
    return render_template('muscle_page.html', workout=muscle_pull_2)


@app.route('/musclePull_1')
def musclePull_1():
    return render_template('muscle_pagen.html', workout=muscle_pull_1)


# @app.route('/musclePull_0')
# def musclePull_0():
#     return render_template('musclepull_0.html',workout=muscle_pull_0)
# ------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/pushWorkout_1_fat')
def pushWorkout_1_fat():
    return Response(shoulderPress_fat(up, 10), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pushWorkout_0_fat')
def pushWorkout_0_fat():
    return Response(chestPress_fat(up, 10), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/fatPush_1')
def fatPush_1():
    return render_template('pushing1.html', workout=muscle_push_1)


@app.route('/fatPush_0')
def fatPush_0():
    return render_template('pushing0.html', workout=muscle_push_0)


@app.route('/pushWorkout_1_muscle')
def pushWorkout_1_muscle():
    return Response(shoulderPress_muscle(up, 8), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pushWorkout_0_muscle')
def pushWorkout_0_muscle():
    return Response(chestPress_muscle(up, 8), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/musclePush_1')
def musclepush_1():
    return render_template('wk1.html', workout=muscle_push_1)


@app.route('/musclePush_0')
def musclepush_0():
    return render_template('wk0.html', workout=muscle_push_0)


# ------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/finish')
def finish():
    return render_template('end.html')


@app.route('/gainMusclePlan')
def gainMusclePlan():
    return render_template('schedule.html')


@app.route('/loseFatPlan')
def loseFatPlan():
    return render_template('process.html')


@app.route('/selection')
def selection():
    return render_template('selection.html')


@app.route('/')
def landing():
    return render_template('point.html')


@app.route('/authenticate')
def hello_world():
    return render_template("login.html")


@app.route('/form_login', methods=['POST', 'GET'])
def login():
    print("hi")
    name1 = request.form['username']
    pwd = request.form['password']
    row = df.loc[df['username'] == name1]
    msg = row['max_speed']
    msg1 = row['average_speed']
    msg2 = row['distance']
    msg3 = row['start_date']
    l = []
    l.append(msg)
    l.append(msg1)
    l.append(msg2)
    l.append(msg3)

    if name1 not in creds:
        return render_template('login.html', info='Invalid User')
    else:
        if creds[name1] != pwd:
            return render_template('login.html', info='Invalid Password')
        else:
            return render_template('home.html', name=name1, details=l[0], speed=l[1], dist=l[2], dt=l[3])


if __name__ == "__main__":
    app.run()

