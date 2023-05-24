import os
from deepface import DeepFace


def my_verify(my_image_path, my_db_path):
    # List of all models
    models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
    # load the Facenet model for face recognition
    model = DeepFace.build_model(models[1])
    # find the person in a photo :
    find_result = DeepFace.find(img_path=my_image_path,
                                db_path=my_db_path,
                                model_name=models[1],
                                enforce_detection=True,
                                detector_backend='retinaface')

    # return the first element of the array which is the dataframe :
    df = find_result[0]

    # transform the dataFrame to a String
    df_string = df.to_string()

    # adding the column person to the Dataframe
    df['Person'] = df['Person'] = [os.path.basename(os.path.dirname(file_path)) for file_path in df['identity']]
    # Iterating over the List of the Dataframes
    print('----------------------------------------------------------------------------')
    # Number of persons found
    num_persons = len(find_result)
    if num_persons == 0:
        print("No person found")
    else:
        print(f"Number of persons found: {num_persons}")
        for df in find_result:
            if not df.empty:
                df['Person'] = [os.path.basename(os.path.dirname(file_path)) for file_path in df['identity']]
                row_with_max_score = df.loc[df['Facenet_cosine'].idxmax()]
                person_with_max_score = row_with_max_score['Person']
                print(f"person identified : {person_with_max_score}")
    print('----------------------------------------------------------------------------')

    # printing the result to the File
    with open('output.csv', 'w') as f:
        for df in find_result:
            f.write(df.to_string(index=False) + '\n\n')


my_verify('img4.jpeg', '/home/reda/Pictures/persons')







