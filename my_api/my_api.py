import os
import datetime

from flask import Flask, jsonify, request
from deepface import DeepFace

app = Flask(__name__)


@app.route('/my-verify', methods=['POST'])
def my_verify():
    # get the image and database paths from the request
    image_path = request.form['image_path']
    db_path = request.form['db_path']

    # load the Facenet model for face recognition
    model = DeepFace.build_model('Facenet')

    # find the person in a photo
    find_result = DeepFace.find(img_path=image_path,
                                db_path=db_path,
                                model_name='Facenet',
                                enforce_detection=False,
                                detector_backend='retinaface')

    # create a list of the names of the identified persons
    persons = []
    for df in find_result:
        if not df.empty:
            df['Person'] = [os.path.basename(os.path.dirname(file_path)) for file_path in df['identity']]
            row_with_max_score = df.loc[df['Facenet_cosine'].idxmax()]
            person_with_max_score = row_with_max_score['Person']
            persons.append(person_with_max_score)

    # create the response data dictionary
    num_persons = len(persons)
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M')

    response_data = {
        "num_persons": num_persons,
        "persons": persons,
        "date": current_date,
        "time": current_time
    }

    # return the response data as JSON
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
