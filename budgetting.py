import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pickled model and scaler
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    sc = pickle.load(f)


# Function to scale input data
def scale_input(input_data):
    scaled_input = sc.transform([input_data])
    return scaled_input


# Function to make prediction
def predict_yield(input_data):
    scaled_input = scale_input(input_data)
    prediction = model.predict(scaled_input)
    return prediction[0]


# Streamlit UI
def main():
    st.title("Budgeting for Agriculture")
    st.write("Enter the values for prediction:")

    # Input boxes for each feature
    rainfall = st.number_input("Rainfall", value=0)
    temperature = st.number_input("Temperature", value=0)
    fertilizer = st.number_input("Fertilizer", value=0)
    seed_cost = st.number_input("Seed Cost", value=0)
    labour_cost = st.number_input("Labour Cost", value=0)
    equipment_cost = st.number_input("Equipment Cost", value=0)
    selling_price_per_unit = st.number_input("Selling Price per Unit", value=0)

    # Calculate total cost
    total_cost = seed_cost + fertilizer + labour_cost + equipment_cost

    if st.button("Predict the yield"):
        input_data = [
            rainfall,
            temperature,
            fertilizer,
            seed_cost,
            labour_cost,
            equipment_cost,
        ]
        prediction = predict_yield(input_data)
        st.write(f"The predicted yield is: {prediction}")

        # Calculate costs based on predicted yield and cost factors
        seed_cost_per_hectare = 200
        fertilizer_cost_per_hectare = 300
        labor_cost_per_hectare = 500
        equipment_cost_per_hectare = 1000

        total_seed_cost = seed_cost_per_hectare * prediction
        total_fertilizer_cost = fertilizer_cost_per_hectare * prediction
        total_labor_cost = labor_cost_per_hectare * prediction
        total_equipment_cost = equipment_cost_per_hectare * prediction

        st.write("Cost breakdown based on predicted yield:")
        st.write(f"Total Seed Cost: {total_seed_cost}")
        st.write(f"Total Fertilizer Cost: {total_fertilizer_cost}")
        st.write(f"Total Labor Cost: {total_labor_cost}")
        st.write(f"Total Equipment Cost: {total_equipment_cost}")

        # Calculate revenue based on predicted yield and selling price
        total_revenue = prediction * selling_price_per_unit
        st.write(f"Total Revenue: {total_revenue}")

        # Calculate profit
        profit = total_revenue - total_cost
        st.write(f"Profit: {profit}")

    # Display total cost
    st.write(f"Total Cost: {total_cost}")


if __name__ == "__main__":
    main()
