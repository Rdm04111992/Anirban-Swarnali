import turtle
import time

screen = turtle.Screen()
screen.bgcolor("lightblue")
screen.title("Sky Writing")

plane = turtle.Turtle()
plane.speed(3)
plane.penup()
plane.goto(-300, 0)
plane.color("gray")

def draw_plane(t):
    t.setheading(0)
    t.shape("triangle")
    t.shapesize(2, 3)

def sky_write(message):
    plane.pensize(3)
    plane.color("white")
    plane.pendown()
    plane.write(message, font=("Arial", 24, "bold"))
    plane.penup()

draw_plane(plane)
time.sleep(1)
plane.goto(-200, 100)
sky_write("I love my Wife Swarnali")

screen.mainloop()