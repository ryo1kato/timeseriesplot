#!/usr/bin/env python2.7
#
# timeseriesplot.py - a matplotlib wrapper library to plot time-series data.
#
#   - Only accepts CSV data when used from commandline.
#   - See '--help' for Useage from command line.
#
# TODO:
#   * moving average
#   * Minor tick labels overlap major ones when using AutoDateMinorLocator
#   * Redraws ticks on resize of plot window for show()
#   * Nicer default window size for show()

VERSION = (0, 1)
USAGE = """%prog [OPTIONS] CSV_FILE[...] [/ CSV_FILE[...]]
  Plot time-series data in CSV_FILE(s) to graph using matplotlib.
  Leftmost column is date-time data: either unix-epoch seconds or
  any string that could be parsed by dateutil.parser.parse() function.

  Existense of non-number argument on 2nd column in the 1st line make
  whole the line into title line. (shown as legend)

  If there's a '/'(slash) in argument, all files on the left to it are plot 
  on the left-Y axis and files on right-Y axis.
  If no '/' given and there's only one file with 3 columns, 2nd column
  is plot on left and 3rd is plot on right-Y axis. When two CSV_FILEs given
  without '/', First file is on left and 2nd file is on right.
  (to prevent this, just put '/' explicitly)
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates, ticker, mlab
from datetime import datetime, date, timedelta
from itertools import cycle
import dateutil.parser
import re
import os


class TimeSeriesData():
    def __init__(self, data_or_array=None, name='', datecol=0, valcol=1):
        self._name = name
        self._dates = []
        self._values = []
        if data_or_array:
            if not isinstance(data_or_array,list):
                array = [data_or_array]
            else:
                array = data_or_array
            for d in array:
                self.append(d, datecol, valcol)

    def _checkdata(self, data):
        if isinstance(data,(list,tuple)) and len(data) >= 2:
            return True
        else:
            raise ValueError("Only accepts list(s) with 2 or more data")

    def _to_datetime(self, datetime_string):
        # FIXME: need to think about TZ.
        try:
            return datetime.utcfromtimestamp( float(datetime_string) )
        except ValueError:
            return dateutil.parser.parse(datetime_string)

    def append(self, data, datecol=0, valcol=1):
        self._dates.append( self._to_datetime(data[datecol]) )
        self._values.append( float(data[valcol]) )

    def dates(self): return self._dates
    def values(self): return self._values
    def name(self): return self._name
    def sort_by_date(self):
        """Need this for line graphs or
        something like moving average or other aggregating functions
        that data data points need to be sorted in time-based order
        """
        if len(self._values) != 0:
            sorted_pair = sorted(zip(self._dates, self._values), key=lambda x: x[0])
            #int((x[0]-y[0]).total_seconds()) )
            self._dates = zip(*sorted_pair)[0]
            self._values = zip(*sorted_pair)[1]


def dummy_data(name, time_interval=300, initval=1000, data_diff_max=100, days=7):
    time_start = 1355000000 #unix epoch-time
    time_end   = time_start + 60*60*24*days # days in sec
    diff_max = 100

    dataset = TimeSeriesData(name=name)
    time = time_start
    val  = initval
    while time < time_end:
        dataset.append(( time, val ))
        time += time_interval
        val += np.random.randint(-diff_max, diff_max)

    return dataset



def timeseriesplot(left, right=None,
                   llabel='', lstyle='', lscale=1, lmin=None, lmax=None,
                   rlabel='', rstyle='-', rscale=1, rmin=None, rmax=None,
                   output=None, moving_average=0, moving_average_style='-'):
    """moving_average: if 0, disable. N>0 indicates average over N data points.
    negative N<0 value indicates moving average over X data points where
    X = len(dataset)/-N. For example, if moving_average=-20, average is taken for
    1/20 = 5% of data points.
    """
    colors=cycle(['b', 'r', 'c', 'm', 'g', 'y', 'k'])
    markers=cycle(['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p',
                   '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])

    def dateplot_data(axis, dataset, linestyle='', marker='AUTO',
                      movavg=0,
                      movavg_style='-'):
        mymarker = marker
        if not isinstance(dataset,list):
            dataset = [dataset]
        for d in dataset:
            if marker == 'AUTO': mymarker = next(markers)
            if linestyle not in ['', ' ']:
                # if data-points are connected, need to sort by date
                # (otherwise, line can go left and right)
                d.sort_by_date()
            axis.plot_date( d.dates(), d.values(), label=d.name(),
                            linestyle=linestyle, marker=mymarker, color=next(colors))
            if movavg != 0 and len(d.values()) > 2:
                if movavg >= 1: # just a number of data points
                    weight = int(movavg) 
                elif movavg > 0: # ratio; 0.05 means 5%
                    weight = int(len( d.values() ) * movavg)
                else: # if it's negative, use 1/weight of data points: -20 means 1/20 = 5%
                    weight = int(len( d.values() ) / abs(movavg))
                if weight < 2: weight = 2
                d.sort_by_date()
                ma = mlab.movavg( d.values(), weight )
                label = "%s (moving avg for %d datapoints)" % (d.name(),weight)
                axis.plot_date( d.dates()[weight-1:], ma, label=label,
                            linestyle=movavg_style,
                            marker=' ', color=next(colors))


    def scale_axis(scale, minval, maxval):
        ## Scale with middle of min and max as pivot
        #mid = (maxval + minval)/2
        #axlength =  (maxval - minval)
        #newmin = mid - axlength/scale
        #newmax = mid + axlength/scale
        #return (newmin, newmax)
        ## Scale with minval as pivot
        return (minval, maxval/scale)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    dateplot_data(ax1, left, linestyle=lstyle,
                  movavg=moving_average, movavg_style=moving_average_style)

    if right:
        # Default right axis is plot with solid line without marker.
        # If no-line style specified, set AUTO marker (like left).
        if rstyle in [ '-', '--', '-.', ':' ]:
            marker=' '
        else:
            marker='AUTO'
        ax2 = ax1.twinx()
        ax2.yaxis.grid(False)
        ax2.set_ylabel(rlabel)
        dateplot_data(ax2, right, linestyle=rstyle, marker=marker)
        ax2.legend(loc=1) # upper right
        ax2.tick_params(axis='both', which='both', labelsize=9)

    # Grid/Locator settings. These must be after *ALL* axis.plot_date() call; otherwise
    # ticks/grid configs are overwritten.
    ax1.set_ylabel(llabel)
    ax1.yaxis.grid(True, color='#8899dd')

    majloc = dates.AutoDateLocator( interval_multiples=True ) 
    majfmt = dates.AutoDateFormatter( majloc )
    majfmt.scaled = {
       365.0    : '%Y',
       30.      : '%b %Y',
       1.0      : '%Y-%m-%d(%a)',
       1./24.   : '%d(%a) %H:%M',
       1./1440. : '%H:%M:%S',
       }

    class AutoDateMinorLocator(dates.AutoDateLocator):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault('minticks', 3)
            kwargs.setdefault('maxticks', 7)
            dates.AutoDateLocator.__init__(self, *args, **kwargs)
        def viewlim_to_dt(self):
            major_ticks = self.axis.get_majorticklocs()
            # to deal with non-uniform interval of major_ticks...
            #like days on month: [1,8,22,29,1,8,...]
            max_major_ticks_interval = max([abs(x2-x1) for (x1, x2) in zip(major_ticks[:-1],major_ticks[1:])])
            return ( dates.num2date(major_ticks[0], self.tz), 
                     dates.num2date(major_ticks[0]+max_major_ticks_interval, self.tz)   )
        def datalim_to_dt(self):
            self.get_view_interval()
            return self.get_view_interval()

    minloc = AutoDateMinorLocator( interval_multiples=True ) 
    #minfmt = dates.AutoDateFormatter( minloc )
    #minfmt.scaled = {
    #   365.0    : '%Y',
    #   30.      : '%b %Y',
    #   1.0      : '%d(%a)',
    #   1./24.   : '%H:%M',
    #   1./1440. : "%H:%M:%S",
    #   }

    ax1.xaxis.set_major_locator( majloc )
    ax1.xaxis.set_major_formatter( majfmt )
    ax1.xaxis.set_minor_locator( minloc )
    #ax1.xaxis.set_minor_formatter( minfmt )


    ax1.xaxis.grid(True, 'major', linewidth=1, linestyle='-',  color='#dd8899')
    ax1.xaxis.grid(True, 'minor', color='#666666')
    ax1.tick_params(axis='both', which='both', labelsize=9)

    # X legend
    ax1.legend(loc=2, numpoints=1) # on upper left, must be after ax1.plot_date()
    for tl in ax1.xaxis.get_majorticklabels() + ax1.xaxis.get_minorticklabels():
        tl.set_rotation(-45)
        tl.set_ha('left')
        tl.set_va('top')
    for tl in ax1.xaxis.get_majorticklabels():
        tl.set_color('red')

    # left axis scale/min/max
    y1min, y1max = ax1.get_ylim()
    if lmin is not None: y1min = lmin
    if lmax is not None: y1max = lmax
    ax1.set_ylim( scale_axis(lscale, y1min, y1max) )

    # right axis scale/min/max
    if right:
        y2min, y2max = ax2.get_ylim()
        if rmin is not None: y2min = rmin
        if rmax is not None: y2max = rmax
        ax2.set_ylim( *scale_axis(rscale, y2min, y2max) )

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.16)
    if output:
        # output format will be auto-detected; see matplotlib manual.
        fig.set_dpi(100)
        fig.set_size_inches(12,6)
        fig.savefig(output, bbox_inches='tight')
    else:
        plt.show()


class TSPlotException(Exception):
    pass

def load_csv(infile, colname_prefix='', colnames=None):
    """load time-series data set from file/fileIO.

    First line can be a column names. If the second column of first line
    is not number (neither int() nor float()-able string), then the
    whole line is assumed to be an column names.

    Number of dataset is auto detected and defined by the first line.
    of input, if subsequent lines has more or less columns, they will be
    either ignored or filled as 0.

    if 'colnames' argument is given, it will be used as column names instead.
    (and also used to define number of columns)
    """

    re_number = re.compile('\d+(.\d+)?')

    if infile == '-':
        fileio = sys.stdin
    elif isinstance(infile,str):
        fileio = open(infile)
    else:
        fileio = infile

    ## try to determine column names and number of columns
    datalines = fileio.readlines()
    try:
        line1_cols = datalines[0].rstrip('\n').split(',')
    except IndexError:
        raise TSPlotException("Empty or broken input")

    nrCol = 2 # this is minimum number of cols we need
    if re_number.match(line1_cols[1]):
        nrCol = len( datalines[0].split(',') )
        colnames = [ colname_prefix + "col-%d"%i for i in xrange(1, nrCol) ]
    else:
        if not colnames:
            colnames = line1_cols[1:] # the first column is time (X-Axis)
        nrCol = len( colnames )
        datalines = datalines[1:]

    if colname_prefix:
        colnames = [ colname_prefix + name for name in colnames ]

    dataset_list = [ TimeSeriesData(name=name) for name in colnames ]

    for l in datalines:
        cols = l.rstrip('\n').split(',')
        time = cols[0]
        data = cols[1:]

        for ds, datum in map(None, dataset_list, data):
            if ds and datum:
                ds.append( (time,datum) )

    return dataset_list

#############################################################################
##
## Command line wrapper for this Python module.
##


def _optparse(args):
    from textwrap import dedent
    import optparse

    epilog_help = dedent("""
    Y-axis options:
      --llabel, --rlabel=LABEL_TEXT
          Left- and right-Y axis label (default is filename)
      --lstyle, --rstyle=STYLE
          Line styles for left- and right-Y axis plot.
          Use matplotlib notation (defaults: ''(no line) and '-')
      --lmin, --lmax, --lscale, --rmin, --rmax, --rscale=VALUE
          Min, max values and scaling factor for left- and right-Y axis.
    """)
    class MyParser(optparse.OptionParser):
        """A parser NOT to strip epilog"""
        def format_epilog(self, formatter):
            return self.epilog
    fmtr = optparse.IndentedHelpFormatter( max_help_position=16 )
    p = MyParser(usage=USAGE, version=("%%prog %d.%d" % VERSION),
                 formatter=fmtr, epilog=epilog_help )
    p.add_option("-o", "--output", metavar="FILENAME", default=None,
            help="Save graph to a file named FILENAME (ex.'mygraph.png')." \
                 "Format is autodetected from extension.")
    p.add_option("-m", "--movavg", "--moving-average", metavar="RANGE", default=0,
            help="Draw moving average for all left-Y axis datapoints." \
                 "Use RANGE number of data points when =>1, '100*RANGE%' of data points if 0<RANGE<1" \
                 "or '-100/RANGE %' of them if RANGE < 0.")

    # FIXME: not implemented yet
    #p.add_option("-c", "--colnames", metavar="NAME1[,NAME2[,...]]",
    #        help="Set names for each columns (when there's multiple files), "
    #             "names are assigned from leftmost column of leftmost file in argument.")

    p.add_option("--llabel", default=None,  help=optparse.SUPPRESS_HELP)
    p.add_option("--rlabel", default=None,  help=optparse.SUPPRESS_HELP)
    p.add_option("--lstyle", default='',    help=optparse.SUPPRESS_HELP)
    p.add_option("--rstyle", default='-',   help=optparse.SUPPRESS_HELP)
    p.add_option("--lscale", type=float, default=1,     help=optparse.SUPPRESS_HELP)
    p.add_option("--rscale", type=float, default=1,     help=optparse.SUPPRESS_HELP)
    p.add_option("--lmin",   type=float, default=None,  help=optparse.SUPPRESS_HELP)
    p.add_option("--lmax",   type=float, default=None,  help=optparse.SUPPRESS_HELP)
    p.add_option("--rmin",   type=float, default=None,  help=optparse.SUPPRESS_HELP)
    p.add_option("--rmax",   type=float, default=None,  help=optparse.SUPPRESS_HELP)

    # FIXME: not implemented yet.
    # p.add_option("-r", "--right", metavar='[FILENUM[,...]][/COLNUM[...]]', default=None,
    #     help="Plot data from FILENAME and/or COLUMN to right axis" \
    #     "'--right=1' will plot all columns from the 1st file in argument to right axis," \
    #     "'--right=/1,2,4' will plot columns 2 and 4 of all the files to right"
    #     "You can specify multiple of this option for complex combinations." )

    return p.parse_args(args)


def main(args):
    (opts, files) = _optparse(args)

    left_dataset = []
    left_label = '(No name)'
    right_files = []
    right_dataset = []
    right_label = '(No name)'

    if len(files) >= 2:
        if '/' in files:
            idx = files.index('/')
            left_files  = files[:idx]
            right_files = files[(idx+1):]
        elif len(files) == 2:
            left_files = [ files[0] ]
            right_files = [ files[1] ]
        else:
            left_files = files
            right_files = []
        for f in left_files:
            if len(left_files) > 1:
                prefix = os.path.basename(f) + ':'
                left_dataset.extend(  load_csv(f, colname_prefix=prefix)  )
            else:
                left_dataset.extend(  load_csv(f)  )
        for f in right_files:
            if len(right_files) > 1:
                prefix = os.path.basename(f) + ':'
                right_dataset.extend(  load_csv(f, colname_prefix=prefix)  )
            else:
                right_dataset.extend(  load_csv(f)  )
        if left_files:    left_label  = os.path.basename( left_files[0] )
        if right_files:   right_label = os.path.basename( right_files[0] )
    else:
        if len(files) == 0:
            left_dataset = load_csv( sys.stdin )
            left_label = '(standard input)'
        else:
            left_dataset = load_csv( files[0] )
            left_label = os.path.basename( files[0] )

        if len(left_dataset) == 2:
            right_dataset = left_dataset[1]
            right_label   = left_label
            left_dataset  = left_dataset[0]

    if opts.llabel: left_label  = opts.llabel
    if opts.rlabel: right_label = opts.rlabel

    optsdict = vars(opts)
    kwarg = { kwd: optsdict[kwd] for kwd in
                [ "lstyle", "rstyle",
                  "lscale", "rscale", "lmin", "lmax",  "rmin",  "rmax"] }

    timeseriesplot(left_dataset, right_dataset,
                output=opts.output,
                moving_average=float(opts.movavg),
                llabel=left_label, rlabel=right_label, **kwarg)


if __name__ == '__main__':
    import sys
    try:
        main( sys.argv[1:] )
    except KeyboardInterrupt:
        pass
    except TSPlotException as e:
        msg = "ERROR: %s\n" % e
        sys.stderr.write(msg)
        sys.exit(1)

