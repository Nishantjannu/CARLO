# CARLO
2D Driving Simulator

CARLO stands for _[CARLA](http://carla.org/) - Low Budget_. CARLO is definitely less realistic than CARLA, but it is much easier to play with. Most importantly, you can easily step the simulator in CARLO, and it is computationally much lighter.
<img width="600" alt="CARLO - Example Image" src="carlo.png" />

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [NumPy](http://www.numpy.org/)
- [TkInter](http://wiki.python.org/moin/TkInter)
- `pip install -e driving-envs`
- `pip install stable_baselines`
- `pip install wandb`
- `pip install tensorflow-gpu`
- `sudo apt-get install ghostcript`: For rendering the env to video.
- `pip install moviepy`

## Running
Simply run
```python
	python example.py
```
for a simple demo. You can also have a look at this file to understand how to customize your own environment. It is very straightforward.

## Features
CARLO currently supports Cars and Pedestrians as the dynamic objects. They both use point-mass dynamics.

It also supports Buildings and Paintings, which are useful mostly for decoration purposes (e.g. sidewalks, zebra crossings, etc).

Collision dynamics are not implemented, but CARLO has methods that can check whether or not there exists a collision between two objects.

There are many hidden features right now. I will reveal them as I start writing a documentation.

## Contributing
Feel free to contribute to the code and/or initiate issues for problems and questions.
