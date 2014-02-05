timeseriesplot.py - a matplotlib wrapper library to plot time-series data.

ABOUT
    An easy-to-use Python module to plot time-series data.
    Comes with neat commandline interface as well.

    The module is capable of plotting multiple dataset in
    left- and right-Y axis, with marker and color auto assigned.
    You can also specify range (min and max value) of axis.


MODULE USAGE EXAMPLE
    Construct TimeSeriesData object and give it to
    timeseriesplot() function.

    Sample1: create empty dataset and then add data one by one
    as read CSV input file and show graph in matplotlib GUI.
        |#!/usr/bin/env python
        |import timeseriesplot.py
        |csvfile = open("myfile.csv")
        |dataset = TimeSeriesData(name="Sample1")
        |for line in csvfile.readlines():
        |    cols = l.rstrip('\n').split(',')
        |    time  = cols[0]
        |    datum = cols[1]
        |    dataset.append( (time,datum) )
        |timeseriesplot( dataset )

   Sample2: first read all the data into table (array of arrays) at once,
   and then give it to constructor of TimeSeriesData().
   Save graph into 'mygraph.png' file in PNG format.
        |#!/usr/bin/env python
        |import timeseriesplot.py
        |csvfile = open("myfile.csv")
        |tabledata = [ linerstrip('\n').split(',') 
        |              for line in csvfile.readlines() ]
        |dataset = TimeSeriesData(tabledata, name="Sample 2")
        |timeseriesplot( dataset, output="mygraph.png" )


COMMANDLINE USAGE EXAMPLE
    $ ./timeseriesplot.py file.csv
        plot data in 'file.csv'. If there is exactly three columns,
        plot 2nd one on left and 3rd one to right-Y axis.

    $ ./timeseriesplot.py file1.csv file2.csv
        plot (all) data in 'file1.csv' on left and 'file2.csv'
        on right-Y axis

    $ ./timeseriesplot.py file1.csv file2.csv file3.csv
        plot (all) data in 'file1.csv' on left-Y axis

    $ ./timeseriesplot.py file1.csv file2.csv / file3.csv
        plot (all) data in 'file1.csv' and 'file2.csv' on left,
        file3.csv to right-Y axis.

    $ ./timeseriesplot.py --heatmap --heatmap-bins=100 file1.csv file2.csv
        plot (all) data in 'file1.csv' as heatmap with 100bins,
        and plot 'file2.csv' on right-Y axis


TODO
    * Unit testing.
    * Timezone handling
    * Fix the support for bar graph.
    * Implement --colnames option

LICENSE
    This script is free to use and/or redistribute under MIT license.
