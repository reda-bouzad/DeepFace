import os
from deepface import DeepFace

# List of all models
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']

# load the Facenet model for face recognition
model = DeepFace.build_model(models[1])

# check if two photos are simular :
verify_result = DeepFace.verify(img1_path="male1.jpg", img2_path="female1.jpg")

# find the person in a photo :
find_result = DeepFace.find(img_path="Messi-Ronaldo-Suarez.jpg",
                            db_path="/home/reda/Pictures/persons",
                            model_name=models[1],
                            enforce_detection=False,
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
print(f"Number of persons found: {num_persons}")
for df in find_result:
    df['Person'] = [os.path.basename(os.path.dirname(file_path)) for file_path in df['identity']]
    row_with_max_score = df.loc[df['Facenet_cosine'].idxmax()]
    person_with_max_score = row_with_max_score['Person']
    print(f"person identified : {person_with_max_score}")
print('----------------------------------------------------------------------------')

# printing the result to the File
with open('output.csv', 'w') as f:
    for df in find_result:
        f.write(df.to_string(index=False) + '\n\n')

