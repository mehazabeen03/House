import streamlit as st
import pickle
import numpy as np

location_mapping = {
    "Poranki": 8,
    "Kankipadu": 5,
    "Benz Circle": 0,
    "Gannavaram": 2,
    "Rajarajeswari Peta": 9,
    "Gunadala": 4,
    "Gollapudi": 3,
    "Enikepadu": 1,
    "Vidhyadharpuram": 10,
    "Penamaluru": 7,
    "Payakapuram": 6
}
status_mapping = {
    "Resale": 2,
    "Under Construction": 3,
    "Ready to move": 1,
    "New": 0
}

direction_mapping = {
    "Not Mentioned": 0,
    "East": 1,
    "West": 3,
    "NorthEast": 2
}

property_type_mapping = {
    "Apartment": 0,
    "Independent Floor": 1,
    "Independent House": 2,
    "Residential Plot": 3
}

with open('Model.pkl','rb') as f:
    model = pickle.load(f)
    
with open('Scaler.pkl','rb') as f:
    scaler = pickle.load(f) 
    
def predict(bed,bath,loc,size,status,face,Type):
    loaction= location_mapping[loc]
    st = status_mapping[status]
    direction  = direction_mapping[face]
    property = property_type_mapping[Type]
    
    input_data = np.array([[bed, bath, loaction, size, st, direction, property]])
    
    input_df = scaler.transform(input_data)
    
    return model.predict(input_df)[0]

if __name__ == '__main__':
    st.header('House Price Prediction')
    
    col1, col2 = st.columns([2, 1])
    
    bed = col1.slider('No of Bedrooms', max_value=10, min_value=0, value=2)
    bath = col1.slider('No of Bathrooms', max_value=10, min_value=0, value=2)
    loc = col1.selectbox('Select a Location', list(location_mapping.keys()))
    size = col1.number_input('Size in Sqft', max_value=10000, min_value=500, value=1000, step=250)
    status = col1.selectbox('Select the Status', list(status_mapping.keys()))
    face = col1.selectbox('Select the Direction', list(direction_mapping.keys()))
    Type = col1.selectbox('Select the Direction', list(property_type_mapping.keys()))
    
    result = predict(bed,bath,loc,size,status,face,Type)
    
    col2.image('https://img.freepik.com/free-photo/blue-house-with-blue-roof-sky-background_1340-25953.jpg', use_column_width=True)
    
    col2.write(f'The predicted value is : {result} Lakhs')