# Crazy-practical

Project of navigating a Crazyflie quadrotor in cluttered environments fully with onboard sensors

## Installation

### cflib typing support

Pylance doesn't seem to be able to resolve the `cflib` package when it is installed as a local package. Creating a symbolic link to the folder gives you full static typing support.

```batch
> mklink /D cflib C:\...\crazyflie-lib-python\cflib
```

## Usage

To run the Crazyflie controller, start the app module.

```
python -m app
```
