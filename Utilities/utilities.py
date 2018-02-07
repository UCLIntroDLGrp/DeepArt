
def selectData(data, nb):
    '''
    Select the first nb examples of the data. Only use for debugging
    :param data: The data we want to select from
    :param nb: The number of examples we want to retain.
    :return: The first nb examples in the dataset
    '''
    if(len(data.shape)==4):
        return data[:nb,:,:,:]
    elif(len(data.shape)==2):
        return data[:nb,:]
    else:
        raise ValueError('data needs to have 4 or 2 dimentions')

def collapseVectors(v):
    '''
    From a probability only retain the highest component as 1 and the rest a 0.
    :param v: Initial probability vector. Needs to be an m*n numpy array with m the number of examples and n the
            number of classes.
    :return: The hot-key vector corresponding to v
    '''

    return (v == v.max(axis=1)[:,None]).astype(int)