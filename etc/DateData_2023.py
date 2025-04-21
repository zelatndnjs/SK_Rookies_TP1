import pandas as pd
from datetime import date, timedelta

start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)
delta = timedelta(days=1)

holidays_2023 = [
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01",
    "2023-05-01",
    "2023-05-05", "2023-05-27", "2023-06-06",
    "2023-08-15",
    "2023-09-28", "2023-09-29", "2023-09-30",
    "2023-10-03", "2023-10-09",
    "2023-12-25"
]
holiday_set = set(pd.to_datetime(holidays_2023))

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

df.to_csv("korean_calendar_2023.csv", index=False)