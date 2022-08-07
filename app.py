import joblib
import streamlit as st
import numpy as np
from prediction import label,predict
import sys
import subprocess
subprocess.check_call([sys.executable,'-m','pip','install','xgboost'])

model=joblib.load('site_eui.sav')

st.set_page_config(page_title='Site Energy Usage Intensity Prediction App',page_icon='⚡',layout='wide')

st.markdown("<h1 style = 'text-align: center;color: #8A03DD;'>Site Energy Usage Intensity Prediction App ⚡</h1>", unsafe_allow_html=True)
st.image('site.png',use_column_width=True)

State_factor=['State_1', 'State_2', 'State_4', 'State_6', 'State_8', 'State_10','State_11']
Building_class=['Commercial', 'Residential']
Facility_type=['Grocery_store_or_food_market','Warehouse_Distribution_or_Shipping_center','Retail_Enclosed_mall', 'Education_Other_classroom','Warehouse_Nonrefrigerated', 'Warehouse_Selfstorage','Office_Uncategorized', 'Data_Center', 'Commercial_Other','Mixed_Use_Predominantly_Commercial','Office_Medical_non_diagnostic', 'Education_College_or_university','Industrial', 'Laboratory','Public_Assembly_Entertainment_culture','Retail_Vehicle_dealership_showroom', 'Retail_Uncategorized','Lodging_Hotel', 'Retail_Strip_shopping_mall','Education_Uncategorized', 'Health_Care_Inpatient','Public_Assembly_Drama_theater', 'Public_Assembly_Social_meeting','Religious_worship', 'Mixed_Use_Commercial_and_Residential','Office_Bank_or_other_financial', 'Parking_Garage','Commercial_Unknown',
'Service_Vehicle_service_repair_shop','Service_Drycleaning_or_Laundry', 'Public_Assembly_Recreation','Service_Uncategorized', 'Warehouse_Refrigerated','Food_Service_Uncategorized', 'Health_Care_Uncategorized','Food_Service_Other', 'Public_Assembly_Movie_Theater','Food_Service_Restaurant_or_cafeteria', 'Food_Sales','Public_Assembly_Uncategorized', 'Nursing_Home','Health_Care_Outpatient_Clinic', 'Education_Preschool_or_daycare','5plus_Unit_Building', 'Multifamily_Uncategorized','Lodging_Dormitory_or_fraternity_sorority','Public_Assembly_Library', 'Public_Safety_Uncategorized','Public_Safety_Fire_or_police_station', 'Office_Mixed_use','Public_Assembly_Other', 'Public_Safety_Penitentiary','Health_Care_Outpatient_Uncategorized', 'Lodging_Other','Mixed_Use_Predominantly_Residential', 'Public_Safety_Courthouse','Public_Assembly_Stadium', 'Lodging_Uncategorized','2to4_Unit_Building', 'Warehouse_Uncategorized']

def main():
    with st.form('prediction_form'):
        st.subheader('Please enter the following information')
        state_factor=st.selectbox("Select the State Factor",options=State_factor)
        building_class=st.selectbox("Select the Building Class",options=Building_class)
        facility_type=st.selectbox("Select the Facility Type",options=Facility_type)
        # Label Encode the above
        building_area=st.text_input("Enter the building area")
        floor_area=st.text_input("Enter the floor area")
        elevation=st.text_input("Enter the elevation of the place")
        energy_star_rating=st.text_input("Enter the energy star rating of the building")
        jan_avg=st.text_input("Enter the average temperature for the month of January")
        feb_avg=st.text_input("Enter the average temperature for the month of February")
        mar_avg=st.text_input("Enter the average temperature for the month of March")
        apr_avg=st.text_input("Enter the average temperature for the month of April")
        may_avg=st.text_input("Enter the average temperature for the month of May")
        jun_avg=st.text_input("Enter the average temperature for the month of June")
        jul_avg=st.text_input("Enter the average temperature for the month of July")
        aug_avg=st.text_input("Enter the average temperature for the month of August")
        sep_avg=st.text_input("Enter the average temperature for the month of September")
        oct_avg=st.text_input("Enter the average temperature for the month of October")
        nov_avg=st.text_input("Enter the average temperature for the month of November")
        dec_avg=st.text_input("Enter the average temperature for the month of December")

        submit=st.form_submit_button("Predict Site EUI")

    if submit:
        state_factor=label(state_factor,State_factor)
        building_class=label(building_class,Building_class)
        facility_type=label(facility_type,Facility_type)

        building_area=float(building_area)
        floor_area=float(floor_area)
        elevation=float(elevation)
        energy_star_rating=float(energy_star_rating)
        jan_avg=float(jan_avg)
        feb_avg=float(feb_avg)
        mar_avg=float(mar_avg)
        apr_avg=float(apr_avg)
        may_avg=float(may_avg)
        jun_avg=float(jun_avg)
        jul_avg=float(jul_avg)
        aug_avg=float(aug_avg)
        sep_avg=float(sep_avg)
        oct_avg=float(oct_avg)
        nov_avg=float(nov_avg)
        dec_avg=float(dec_avg)

        win_avg=(dec_avg+jan_avg+feb_avg)/3.0
        spr_avg=(mar_avg+apr_avg+may_avg)/3.0
        sum_avg=(jun_avg+jul_avg+aug_avg)/3.0
        aut_avg=(sep_avg+oct_avg+nov_avg)/3.0

        data=np.array([floor_area,energy_star_rating,elevation,jan_avg,feb_avg,mar_avg,
        apr_avg,may_avg,jun_avg,jul_avg,aug_avg,sep_avg,
        oct_avg,nov_avg,dec_avg,state_factor,building_class,facility_type,building_area,
        win_avg,spr_avg,sum_avg,aut_avg]).reshape(1,-1)

        pred=predict(data=data,model=model)

        st.write("The Site Energy Usage Intensity is: {} kBtu/year".format(pred[0]))

if __name__=='__main__':
    main()