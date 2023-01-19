# Serving ML Model via Flask

## Flaskrml

Demonstrating how to serve ML models via REST using flask 

## Setting up the Backend

### Install Dependencies

1. **Python 3.10** - Follow instructions to install the latest version of python for your platform in the [python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)

2. **Virtual Environment** - Recommend working within a virtual environment . This keeps your dependencies for each project separate and organized. Instructions for setting up a virual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

3. **PIP Dependencies** - Once your virtual environment is setup and running, install the required dependencies by navigating to the `/backend` directory and running:

```bash
pip install -r requirements.txt
```

#### Key Pip Dependencies

- [Flask](http://flask.pocoo.org/) is a lightweight backend microservices framework. Flask is required to handle requests and responses.

- [Trax](https://github.com/google/trax) Trax is an end-to-end library for deep learning that focuses on clear code and speed. I

- [Tensorflow](https://www.tensorflow.org/) An end-to-end machine learning platform

- [jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) JAX is NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research.

### Run the Server


#### Backend
To run the application navigate to the backend folder and  run the following commands: 
```
export FLASK_APP=app
export FLASK_ENV=development
flask run
```

These commands put the application in development and directs our application to use the `__init__.py` file in our flaskr folder. Working in development mode shows an interactive debugger in the console and restarts the server whenever changes are made. If running locally on Windows, look for the commands in the [Flask documentation](http://flask.pocoo.org/docs/1.0/tutorial/factory/).

The application is run on `http://127.0.0.1:5000/` by default and is a proxy in the frontend configuration. 

