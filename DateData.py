import pandas as pd
from datetime import date, timedelta

start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)
delta = timedelta(days=1)

holidays_2024 = [
    "2024-01-01",
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-03-01",
    "2024-04-10",
    "2024-05-01",
    "2024-05-05", "2024-05-06",
    "2024-06-06",
    "2024-08-15",
    "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19",
    "2024-10-03",
    "2024-10-09",
    "2024-12-25"
]
holiday_set = set(pd.to_datetime(holidays_2024))

dates = []
weekdays = []
is_holidays = []

current = start_date
while current <= end_date:
    dates.append(current)
    weekdays.append(current.weekday())
    is_holidays.append(1 if pd.to_datetime(current) in holiday_set else 0)
    current += delta

df = pd.DataFrame({
    'date': dates,
    'weekday': weekdays,
    'is_holiday': is_holidays
})

df.to_csv("korean_calendar_2024.csv", index=False)