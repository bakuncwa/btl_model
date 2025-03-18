import random
import csv
from datetime import datetime, timedelta

skincare_categories = [
    "CLEANSERS", "TONERS", "EXFOLIANTS", "MOISTURIZERS", "SERUMS",
    "EYECARE", "MASKS", "SUNSCREENS", "ANTIAGING"
]

weather_conditions = [
    "Clear", "Cloudy", "Intermittent clouds", "Mostly clear",
    "Mostly cloudy w showers", "Mostly cloudy with storms",
    "Mostly sunny", "Partly cloudy", "Partly sunny",
    "Partly sunny with showers", "Rain", "Showers",
    "Sunny", "Thunderstorms"
]

product_code_map = {category: set() for category in skincare_categories}

for category in skincare_categories:
    while len(product_code_map[category]) < 30:
        product_code_map[category].add(f"SK{random.randint(1000, 9999)}")

product_code_map = {category: list(codes) for category, codes in product_code_map.items()}

data = []
num_records = 20000

for _ in range(num_records):
    user_id = random.randint(1000, 5000)
    age = random.randint(18, 65)

    transaction_id = random.randint(100000, 999999)
    transaction_date = (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))).strftime("%Y-%m-%d")

    category_name = random.choice(skincare_categories)
    category_id = skincare_categories.index(category_name) + 1

    product_code = random.choice(product_code_map[category_name])

    price = round(random.uniform(5.0, 200.0), 2)
    quantity = random.randint(1, 5)
    rating = round(random.uniform(1.0, 5.0), 1)

    temperature = round(random.uniform(-10.0, 40.0), 1)
    humidity = round(random.uniform(10.0, 90.0), 1)
    wind_speed = round(random.uniform(0.0, 30.0), 1)

    weather = random.choice(weather_conditions)
    realfeel = round(temperature + random.uniform(-3.0, 3.0), 1)

    data.append([
        user_id, age, price, quantity, rating, transaction_id, transaction_date, product_code,
        category_id, category_name, temperature, humidity, wind_speed, weather, realfeel
    ])

csv_file = "skincare_transactions.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "user_id", "age", "price", "quantity", "rating", "transaction_id", "transaction_date",
        "product_code", "category_id", "category_name", "temperature", "humidity", "wind_speed",
        "weather_list", "realfeel_list"
    ])
    writer.writerows(data)

print(f"Sample skincare transaction data with {num_records} records saved to {csv_file}")