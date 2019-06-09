# Respiratory mechanics, work of breathing and diaphragm EMG entropy calculations
------------------------------------------------------------------------ 
(c) Copyright 2019 Emil Schwarz Walsted <emilwalsted@gmail.com>
This is a rewrite of my MATLAB code from 2016, with some changes/additions.
The latest version of this code is available at GitHub: ...

This code calculates respiratory mechanics and/or in- and expiratory work of 
breathing and/or diaphragm EMG entropy from a time series of 
pressure/flow/volume recordings obtained  e.g. from LabChart.

Currently, supported input formats are: MATLAB, Excel and CSV. You can extend
this yourself by modifying the 'load()' function.

For usage, please see the example file 'example.py'.

*** Note to respiratory scientists: ***
I created this code for the purpose of my own 
work and shared it hoping that someone else might find it useful. If you have 
any suggestions that will make the code more useful to you or generally, please 
email me to discuss.

*** Note to software engineers/computer scientists: ***
I realise that this code is not as concise or computationally efficient as it 
could be. I have focused on readability in an attempt to enable respiratory 
scientists with basic programming skills to understand, debug and modify/extend
the code themselves.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
