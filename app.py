import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.title("Hệ Thống Phát Hiện Gian Lận Giao Dịch")
st.write("Nhập thông tin giao dịch vào các trường bên dưới:")

merchant = st.text_input("Merchant Name (Tên đơn vị chấp nhận thanh toán)")
category = st.text_input("Category (Danh mục giao dịch)")
amt = st.number_input("Transaction Amount (Số tiền giao dịch)", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude (Vĩ độ khách hàng)", format="%.6f")
long = st.number_input("Longitude (Kinh độ khách hàng)", format="%.6f")
merch_lat = st.number_input("Merchant Latitude (Vĩ độ của đơn vị chấp nhận thanh toán)", format="%.6f")
merch_long = st.number_input("Merchant Longitude (Kinh độ của đơn vị chấp nhận thanh toán)", format="%.6f")
hour = st.slider("Transaction Hour (Giờ giao dịch)", 0, 23, 12)
day = st.slider("Transaction Day (Ngày giao dịch)", 1, 31, 15)
month = st.slider("Transaction Month (Tháng giao dịch)", 1, 12, 6)
gender = st.selectbox("Gender (Giới tính)", ["Male", "Female"])
cc_num = st.text_input("Credit Card number (Số thẻ tín dụng)")

distance = haversine(lat, long, merch_lat, merch_long)

if st.button("Kiểm tra gian lận"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, cc_num, hour, day, month, gender, distance]],
                          columns=['merchant', 'category', 'amt', 'cc_num', 'hours', 'day', 'month', 'gender', 'distance'])

        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        
        
        result = "💳 Giao dịch **gian lận**!" if prediction == 1 else "✅ Giao dịch **hợp lệ**."
        st.subheader(f"Kết quả dự đoán: {result}")
    else:
        st.error("Vui lòng điền đầy đủ các trường yêu cầu.")
