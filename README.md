## Requirements

Before starting, make sure you have:

* **Python 3.10**
* **Git**
* A working internet connection for installing Python dependencies

Python 3.10 is recommended because the project dependencies were developed/tested around this version.

---

## 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/adityaghoradpersonal/AI-Fitness-Trainer.git
```

Then enter the project directory:

```bash
cd AI-Fitness-Trainer
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment so the project's dependencies do not interfere with other Python projects.

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3.10 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, your terminal should show something similar to:

```text
(.venv)
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If installation fails because of a package/version issue, make sure you are using **Python 3.10** and that the virtual environment is activated.

---

## 4. Run the Application

From the project root:

```bash
streamlit run main.py
```

Streamlit will provide a local address, normally:

```text
http://localhost:8501
```

Open that address in your browser.

---

## 5. Run the AI Demo

The project includes a demo exercise video:

```text
assets/videos/demo_2.mp4
```

The application automatically loads this video.