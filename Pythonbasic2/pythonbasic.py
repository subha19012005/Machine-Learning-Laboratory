import csv
from datetime import datetime

VALID_ACTIVITIES = ["LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"]

# -----------------------------
# Student class
# -----------------------------
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []  # list of (activity, date, time)

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def get_summary(self):
        logins = 0
        submissions = 0
        for act, _, _ in self.activities:
            if act == "LOGIN":
                logins += 1
            elif act == "SUBMIT_ASSIGNMENT":
                submissions += 1
        return logins, submissions

# -----------------------------
# Helper functions
# -----------------------------
def is_valid_student_id(student_id):
    return student_id.startswith("S") and student_id[1:].isdigit()

def is_valid_activity(activity):
    return activity in VALID_ACTIVITIES

def parse_date_time(date_str, time_str):
    try:
        dt = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M")
        return dt.date(), dt.time()
    except:
        return None, None

# -----------------------------
# Read log file
# -----------------------------
def read_log_file(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter="|")
        for line_number, row in enumerate(reader, 1):
            if len(row) != 5:
                print(f"Skipping line {line_number}: wrong number of columns")
                continue

            student_id = row[0].strip()
            name = row[1].strip()
            activity = row[2].strip()
            date_str = row[3].strip()
            time_str = row[4].strip()

            if not is_valid_student_id(student_id):
                print(f"Skipping line {line_number}: invalid student ID")
                continue
            if not is_valid_activity(activity):
                print(f"Skipping line {line_number}: invalid activity")
                continue

            date, time = parse_date_time(date_str, time_str)
            if date is None:
                print(f"Skipping line {line_number}: invalid date/time")
                continue

            yield student_id, name, activity, date, time

# -----------------------------
# Process logs and generate report
# -----------------------------
def process_logs(input_file, output_file):
    students = {}
    login_status = {}  # track if a student is logged in

    daily_stats = {}  # track daily counts

    for student_id, name, activity, date, time in read_log_file(input_file):
        if student_id not in students:
            students[student_id] = Student(student_id, name)

        students[student_id].add_activity(activity, date, time)

        # Daily stats
        if date not in daily_stats:
            daily_stats[date] = {"LOGIN":0, "LOGOUT":0, "SUBMIT_ASSIGNMENT":0}
        daily_stats[date][activity] += 1

        # Abnormal login check
        if activity == "LOGIN":
            if login_status.get(student_id, False):
                print(f"Abnormal login detected: {student_id} on {date} at {time}")
            login_status[student_id] = True
        elif activity == "LOGOUT":
            login_status[student_id] = False

    # Write report
    with open(output_file, "w") as f:
        f.write("StudentID | Name | Total Logins | Total Submissions\n")
        print("StudentID | Name | Total Logins | Total Submissions")
        for s in students.values():
            logins, submissions = s.get_summary()
            line = f"{s.student_id} | {s.name} | {logins} | {submissions}"
            print(line)
            f.write(line + "\n")

        f.write("\nDaily Activity Stats:\n")
        print("\nDaily Activity Stats:")
        for date, stats in daily_stats.items():
            line = f"{date}: LOGIN={stats['LOGIN']}, LOGOUT={stats['LOGOUT']}, SUBMIT_ASSIGNMENT={stats['SUBMIT_ASSIGNMENT']}"
            print(line)
            f.write(line + "\n")

# -----------------------------
# Run program
# -----------------------------
if __name__ == "__main__":
    process_logs("student_logs.csv", "student_report.txt")