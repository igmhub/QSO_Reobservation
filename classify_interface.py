from classify import *
from optparse import OptionParser
from astropy.table import Table
import numpy as np
from keras.utils import np_utils
def options():
    # Options
    parser = OptionParser()

    parser.add_option('--input-path', dest='input_path', default=None,
                help='Input path to brick files', type='string')
    parser.add_option('--brick-name', dest='brick_name', default=None,
                help='Name of the bricks to read', type='string')
    parser.add_option('--output-name', dest='output_name', default=None,
                help='Path the output table', type='string')
    parser.add_option('--truth-table', dest='truth_table', default=None,
                help='Path to the input truth table', type='string')
    parser.add_option('--table-rm', dest='table_rm', default=None,
                help='Path to redmonster output', type='string')
    parser.add_option('--num-train', dest='numtrain', default=1000,
                help='Number of training samples', type='int')

    (o, args) = parser.parse_args()
    return o,args
def main():
    # Get the data from the bricks
    o, args = options()
    hdus = readBricks(o.input_path,o.brick_name)
    nspec = hdus[0][0].data.shape[0]
    flux_b = downsample(hdus,0,nspec,20)
    flux_r = downsample(hdus,1,nspec,23)
    flux_z = downsample(hdus,2,nspec,35,si=3)
    qso_flux = np.hstack([flux_b,flux_r,flux_z])
    # Let's read the truth table_train
    truth_tab = Table.read(o.truth_table)
    # Let's read redmonster's output (if exists)
    try:
        rm_tab = Table.read(o.table_rm)
    except:
        print 'No input to redmonster found, continuing without it.'
    #Prepare the data
    yvec = encode_truth(truth_tab)
    #If we have outputs from redmonster, include that information
    if o.table_rm is not None:
        xvec = np.zeros((qso_flux.shape[0],qso_flux.shape[1]+2))
        xvec[:,-2]=encode_rm(rm_tab)
        xvec[:,-1]=rm_tab['Z']
    else:
        xvec = qso_flux
    # Randomly select a trining sample
    selected_training = np.random.choice(np.arange(len(truth_tab)),size=o.numtrain)
    # Train the classifier
    predicted_output = quassifier_run(xvec[selected_training],yvec[selected_training],xvec)
    table_results = Table([Table(hdus[0][4].data)['TARGETID'],Table(hdus[0][4].data)['MAG'],predicted_output],names=('TARGETID','MAG','PROB'))
    table_results.write(o.output_name)

if __name__=="__main__":
    main()
