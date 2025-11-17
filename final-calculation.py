import joblib
import pandas as pd

model = joblib.load("sleep_model.pkl")

def time_to_minutes(t):
    h, m = map(int, t.split(":"))
    return h * 60 + m

def minutes_to_time(m):
    h = (m // 60) % 24
    m = m % 60
    return f"{h:02d}:{m:02d}"

# Get user inputs
print("=== Sleep Predictor ===\n")
bedtime = input("Enter bedtime (HH:MM): ")
screen_time = int(input("Enter screen time before bed (minutes): "))
stress = int(input("Enter stress level (1-10): "))
weather = input("Enter weather (e.g., sunny, rainy, cloudy): ")
alarms = int(input("Enter number of alarms set: "))
desired_wake = input("Enter desired wake time (HH:MM): ")

features = pd.DataFrame([{
    "bedtime": time_to_minutes(bedtime),
    "screen_time": screen_time,
    "stress": stress,
    "weather": weather,
    "alarms": alarms
}])

pred_hours = model.predict(features)[0]

# Calibrate predictions to be more conservative
pred_hours = max(4.0, pred_hours * 0.80 - 1.0)  # Stronger reduction

print("\n=== Results ===")
print("Predicted sleep duration:", round(pred_hours, 2), "hours")

pred_wake = time_to_minutes(bedtime) + int(pred_hours * 60)

# Handle crossing midnight
if pred_wake >= 1440:  # 24 hours in minutes
    pred_wake -= 1440

pred_wake_time = minutes_to_time(pred_wake)

print("Predicted wake up time:", pred_wake_time)

desired_wake_minutes = time_to_minutes(desired_wake)

if pred_wake > desired_wake_minutes:
    print("⚠️ Oversleep likely!")
else:
    print("✅ You will wake before your desired time.")
