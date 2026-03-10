#                  
from datetime import datetime, timedelta


def increment_day(date, NoLeapYear=False ):
    # Create a datetime object from the date dictionary
    current_date = datetime(date[0], date[1], date[2])
    
    # Increment the day using timedelta
    next_date = current_date + timedelta(days=1)

    # Manually adjust if next_date is February 29 and NoLeapYear is True
    if (NoLeapYear==True) and (next_date.month == 2) and (next_date.day == 29):
        # Adjust to March 1
        next_date = datetime(next_date.year, 3, 1)

    # Update the date dictionary with the new date
    date[0], date[1], date[2] = next_date.year, next_date.month, next_date.day
    return date

def increment_hours(date, nhours=1, NoLeapYear=False ):
    # Create a datetime object from the date dictionary
    current_date = datetime(date[0], date[1], date[2], date[3])
    
    # Increment the day using timedelta
    next_date = current_date + timedelta(hours=nhours)

    # Manually adjust if next_date is February 29 and NoLeapYear is True
    if (NoLeapYear==True) and (next_date.month == 2) and (next_date.day == 29):
        # Adjust to March 1
        next_date = datetime(next_date.year, 3, 1)

    # Update the date dictionary with the new date
    date[0], date[1], date[2] , date[3] = next_date.year, next_date.month, next_date.day, next_date.hour
    return date

def increment_month(date):
    # Increment the month and handle month/year change if needed
    # This is a simplistic implementation and does not handle all edge cases
    date[1] += 1
    if date[1] > 12:
        date[1] = 1
        date[0] += 1
    return date

def date_str(date):
    year=date[0]
    if len(date)==2:
        month=date[1]
    elif len(date)==3:
        month=date[1]
        day=date[2]
        if day == 99:
            day='*'
    elif len(date)==4:
        month=date[1]
        day=date[2]
        hour=date[3]
    
    date_=f"{year:04d}-{month:02d}-{day:02d}-{hour*3_600:05d}"
