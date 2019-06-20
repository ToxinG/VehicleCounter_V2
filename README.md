# VehicleCounter_V2


You may pause/unpause playing video with space and stop with esc. When paused, you may watch it frame-by-frame with any key (except for space and esc, of course).

## June, 20th, 2019.

Added option to log videos to a database. This program uses a database on the localhost but does not create it, so you'll need to do something with it to use this option.

Some minor fixes and code style changes.

And a strange "feature" appeared without any visible reasons. When a car leaves the screen, the frame which was tracking does not disappear until another car goes there. It happens during logging. And I even tried to run the first version which definitely did not have this problem. Now it does))))))))) The only significant thing that changed since the time when there was not such problem is that I installed Postgres and created the database used by the new version. But the old one does not interact with databases at all. It looks like some dark magic. I even tried rebooting my computer.
