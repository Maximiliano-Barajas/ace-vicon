from flask import Flask, render_template, request, redirect, url_for, session, flash
from model import compute_similarity, load_csv, build_reference_model
import matplotlib.pyplot as plt
import os
import json
import hashlib
import sqlite3
import datetime

app = Flask(__name__)
app.secret_key = "tennis_secret_key_change_in_production"

UPLOAD_FOLDER = "uploads"
USERS_FILE = "users.json"
DB_FILE = "tennisiq.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ IMPORTANT: FIXED PATHS
REFERENCE_FILES = [
    "data/max_serves/max1.csv",
    "data/max_serves/max2.csv",
    "data/max_serves/max3.csv",
    "data/max_serves/max4.csv",
    "data/max_serves/max5.csv",
]

# ── DATABASE ─────────────────────────────────────────────


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            filename TEXT NOT NULL,
            player_key TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_style TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def save_session(username, filename, player_key, player_name, player_style, score):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO sessions (username, filename, player_key, player_name, player_style, score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            username,
            filename,
            player_key,
            player_name,
            player_style,
            score,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()
    conn.close()


def get_user_sessions(username):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT * FROM sessions WHERE username = ?
        ORDER BY created_at DESC
    """,
        (username,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


init_db()

# ── AUTH ─────────────────────────────────────────────


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ── FEEDBACK ─────────────────────────────────────────────

PLAYER_FEEDBACK = {
    "max": {
        "name": "Max (Reference Player)",
        "style": "Model Serve",
        "tips": [
            "Your serve is being compared to Max's recorded Vicon motion data.",
            "Focus on matching timing and trajectory.",
            "Differences in motion path will reduce your similarity score.",
            "Work on consistency across your swing.",
        ],
    }
}

# ── GRAPH FUNCTION ─────────────────────────────────────────


def create_plot(user_file):
    user = load_csv(user_file)
    mean, _ = build_reference_model(REFERENCE_FILES)

    min_len = min(len(user), len(mean))
    user = user[:min_len]
    mean = mean[:min_len]

    plt.figure()
    plt.plot(user[:, 0], label="User")
    plt.plot(mean[:, 0], label="Reference")
    plt.legend()

    plot_path = os.path.join("static", "plot.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# ── ROUTES ───────────────────────────────────────────────


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()

        if username in users and users[username] == hash_password(password):
            session["user"] = username
            return redirect(url_for("home"))

        flash("Invalid username or password")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]

        if password != confirm:
            flash("Passwords do not match")
            return render_template("register.html")

        users = load_users()

        if username in users:
            flash("Username already taken")
            return render_template("register.html")

        users[username] = hash_password(password)
        save_users(users)

        session["user"] = username
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/analyse")
def analyse():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("analyse.html", user=session["user"])


@app.route("/myprogress")
def myprogress():
    if "user" not in session:
        return redirect(url_for("login"))

    sessions = get_user_sessions(session["user"])

    chart_sessions = list(reversed(sessions[:10]))
    chart_labels = [s["created_at"][5:10] for s in chart_sessions]
    chart_scores = [s["score"] for s in chart_sessions]

    avg_score = (
        round(sum(s["score"] for s in sessions) / len(sessions), 1) if sessions else 0
    )
    best_score = max((s["score"] for s in sessions), default=0)

    return render_template(
        "myprogress.html",
        user=session["user"],
        sessions=sessions,
        chart_labels=json.dumps(chart_labels),
        chart_scores=json.dumps(chart_scores),
        avg_score=avg_score,
        best_score=best_score,
        total_sessions=len(sessions),
    )


# ── MAIN ANALYSIS ─────────────────────────────────────────


@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    file = request.files.get("media")

    if not file or file.filename == "":
        flash("No file uploaded.")
        return redirect(url_for("analyse"))

    if not file.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.")
        return redirect(url_for("analyse"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    try:
        score = compute_similarity(save_path, REFERENCE_FILES)
        score = round(score, 1)

        plot_path = create_plot(save_path)

    except Exception as e:
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for("analyse"))

    feedback = PLAYER_FEEDBACK["max"]

    save_session(
        username=session["user"],
        filename=file.filename,
        player_key="max",  # ✅ FIXED
        player_name=feedback["name"],
        player_style=feedback["style"],
        score=score,
    )

    return render_template(
        "result.html",
        user=session["user"],
        filename=file.filename,
        player=feedback,
        score=score,
        plot_path=plot_path,
    )


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ── RUN ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)
