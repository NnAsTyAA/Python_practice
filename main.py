import numpy as np
import imageio
from vispy import app, scene
from vispy.geometry import Rect
from funcs import init_boids, directions, propagate, flocking, circle_init
app.use_app('pyglet')

w, h = 1920, 1088
N, ci = 5000, 5
dt = 0.1
asp = w / h
perception = 0.01
frames_count = 2300
count = 0
better_walls_w = 0.05
vrange=(0, 0.01)
arange=(0, 0.005)

#                    c      a    s      w
coeffs = np.array([0.33, 1.22, 0.23, 0.3, 0.01])

boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

circle_arr = np.zeros((ci, 3), dtype=np.float64)
circle_init(circle_arr, ci, asp)

arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)

circle = [scene.visuals.Ellipse(center=circle_arr[i, :2],
                                color='#FF000000',
                                border_width=2,
                                border_color='r',
                                radius=[circle_arr[i, 2], circle_arr[i, 2]],
                                parent=view.scene) for i in range(ci)]

t1 = scene.visuals.Text(parent=canvas.scene, color='red')
t1.pos = 18 * canvas.size[0] // 20, canvas.size[1] // 35
t1.font_size = 10

t2 = scene.visuals.Text(parent=canvas.scene, color='red')
t2.pos = 18 * canvas.size[0] // 20, canvas.size[1] // 8
t2.font_size = 10
info = f"Boids number: {N}\n"
info += f"""a: {coeffs[1]}
 c: {coeffs[0]}
 s: {coeffs[2]}
 w: {coeffs[3]}
 n: {coeffs[4]}"""
t2.text = info

imw = imageio.get_writer(f'visualisation_2_{N}.mp4', fps=60)


def create_video(event):
    """
    Recording video visualization of agent interaction
    :param event:
    :return: None
    """
    global count

    if count % 60 == 0:
        t1.text = f"fps: {canvas.fps}"
        print(count)

    count += 1

    flocking(boids, circle_arr, perception, coeffs, asp, vrange, better_walls_w)
    propagate(boids, dt, vrange, arange)
    arrows.set_data(arrows=directions(boids, dt))

    if count <= frames_count:
        fr = canvas.render(alpha=False)
        imw.append_data(fr)
    else:
        imw.close()
        app.quit()


def update(event):
    """
    Visualization of agent interaction
    :param event:
    :return: None
    """
    global count

    if count % 60 == 0:
        t1.text = f"fps: {canvas.fps}"

    count += 1

    flocking(boids, circle_arr, perception, coeffs, asp, vrange, better_walls_w)
    propagate(boids, dt, vrange, arange)
    arrows.set_data(arrows=directions(boids, dt))
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=create_video)
    canvas.measure_fps()
    app.run()
