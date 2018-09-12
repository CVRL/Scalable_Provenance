import cv2, numpy, math, os

# Auxiliar functions
def _numpy1_9Unique(ar, return_index=False, return_inverse=False, return_counts=False):
    """
    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements: the indices of the input array
    that give the unique values, the indices of the unique array that
    reconstruct the input array, and the number of times each unique value
    comes up in the input array.
    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique value comes up
        in `ar`.
        .. versionadded:: 1.9.0
    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.
        .. versionadded:: 1.9.0
    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.
    Examples
    --------
    >>> numpy.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = numpy.array([[1, 1], [2, 3]])
    >>> numpy.unique(a)
    array([1, 2, 3])
    Return the indices of the original array that give the unique values:
    >>> a = numpy.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = numpy.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')
    Reconstruct the input array from the unique values:
    >>> a = numpy.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = numpy.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])
    """
    ar = numpy.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (numpy.empty(0, numpy.bool),)
            if return_inverse:
                ret += (numpy.empty(0, numpy.bool),)
            if return_counts:
                ret += (numpy.empty(0, numpy.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = numpy.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = numpy.cumsum(flag) - 1
            inv_idx = numpy.empty(ar.shape, dtype=numpy.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = numpy.concatenate(numpy.nonzero(flag) + ([ar.size],))
            ret += (numpy.diff(idx),)
    return ret

def __range(a, bins):
    '''Compute the histogram range of the values in the array a according to
    scipy.stats.histogram.'''
    a = numpy.asarray(a)
    a_max = a.max()
    a_min = a.min()
    s = 0.5 * (a_max - a_min) / float(bins - 1)
    return (a_min - s, a_max + s)

def __entropy(data):
    '''Compute entropy of the flattened data set (e.g. a density distribution).'''
    # normalize and convert to float
    data = data / float(numpy.sum(data))
    
    # for each grey-value g with a probability p(g) = 0,
    # the entropy is defined as 0,
    # therefore we remove these values and also flatten the histogram
    data = data[numpy.nonzero(data)]
    
    # compute entropy
    return -1. * numpy.sum(data * numpy.log2(data))

# Mutual information calculation
def _computeMutualInformation(i1, i2, bins=256):
    """
    Computes the mutual information (MI) (a measure of entropy) between two images.
    MI is not real metric, but a symmetric and nonnegative similarity measures that
    takes high values for similar images. Negative values are also possible.
    Intuitively, mutual information measures the information that X and Y share: it
    measures how much knowing one of these variables reduces uncertainty about the other.
    The Entropy is defined as:
    \f[
        H(X) = - \sum_i p(g_i) * ln(p(g_i)
    \f]
    with \f$p(g_i)\f$ being the intensity probability of the images grey value \f$g_i\f$.
    Assuming two image R and T, the mutual information is then computed by comparing the
    images entropy values (i.e. a measure how well-structured the common histogram is).
    The distance metric is then calculated as follows:
    \f[
        MI(R,T) = H(R) + H(T) - H(R,T) = H(R) - H(R|T) = H(T) - H(T|R)
    \f]
    A maximization of the mutual information is equal to a minimization of the joint
    entropy.
    Note
    ----
    @see medpy.filter.ssd() for another image metric.
    @param i1 the first image
    @type i1 array-like sequence
    @param i2 the second image
    @type i2 array-like sequence
    @param bins the number of histogram bins (squared for the joined histogram)
    @type bins int
    @return the mutual information distance value
    @rtype float
    """
    # pre-process function arguments
    i1 = numpy.asarray(i1)
    i2 = numpy.asarray(i2)
        
    # compute i1 and i2 histogram range
    i1_range = __range(i1, bins)
    i2_range = __range(i2, bins)
    
    if (i1_range[1] - i1_range[0] > 1.0 and
        i2_range[1] - i2_range[0] > 1.0):        
        # compute joined and separated normed histograms
        i1i2_hist, _, _ = numpy.histogram2d(i1.flatten(), i2.flatten(), bins=bins,
                                         range=[i1_range, i2_range])  # Note: histogram2d does not flatten array on its own
        i1_hist, _ = numpy.histogram(i1, bins=bins, range=i1_range)
        i2_hist, _ = numpy.histogram(i2, bins=bins, range=i2_range)
        
        # compute joined and separated entropy
        i1i2_entropy = __entropy(i1i2_hist)
        i1_entropy = __entropy(i1_hist)
        i2_entropy = __entropy(i2_hist)
        
        # compute and return the mutual information distance
        return i1_entropy + i2_entropy - i1i2_entropy
    
    else:
        return 0.0

# MSE
def _computeMSE(i1, i2):
    width = i1.shape[1]
    height = i1.shape[0]
    channelCount = i1.shape[2]
    
    i1 = cv2.resize(i1.astype(numpy.float), (width, height))
    i2 = cv2.resize(i2.astype(numpy.float), (width, height))
    
    mse = numpy.power(i1 - i2, 2)
    mse = numpy.sum(mse) / (width * height * channelCount)
    
    return mse
        
# Color histogram match.
def _colorHistMatch(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: numpy.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: numpy.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: numpy.ndarray
            The transformed output image
    """
    
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = _numpy1_9Unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = _numpy1_9Unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = numpy.cumsum(s_counts).astype(numpy.float64)
    t_quantiles = numpy.cumsum(t_counts).astype(numpy.float64)
    
    if len(s_quantiles) > 0 and len(t_quantiles) > 0:
        s_quantiles /= s_quantiles[-1]    
        t_quantiles /= t_quantiles[-1]
    
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = numpy.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(oldshape)
    
    else:
        return source

# Transforms image 1 towards image 2    
def _transform(image1, keypoints1, image2, keypoints2, minImageSize):
    keypointCount = numpy.min((len(keypoints1), len(keypoints2)))
    points1 = []
    points2 = []
    
    for i in range(keypointCount):
        for j in range(keypointCount):
            points1.append([[keypoints1[j].pt[0], keypoints1[j].pt[1]]])
            points2.append([[keypoints2[j].pt[0], keypoints2[j].pt[1]]])
    
    x1, y1, w1, h1 = cv2.boundingRect(numpy.array(points1).astype(numpy.int32))
    x2, y2, w2, h2 = cv2.boundingRect(numpy.array(points2).astype(numpy.int32))
    
    # patches
    patch1 = image1[y1:y1 + h1, x1:x1 + w1]
    patch2 = image2[y2:y2 + h2, x2:x2 + w2]
    
    if (patch1.shape[0] < minImageSize or patch1.shape[1] < minImageSize or 
        patch2.shape[0] < minImageSize or patch2.shape[1] < minImageSize):
        return None, None
    
    # transforms patch1 into patch2, and returns
    patch1 = _colorHistMatch(patch1, patch2)
    patch1 = cv2.resize(patch1, (patch2.shape[1], patch2.shape[0]))
    return patch1, patch2

def getSimilarity(image1, keypoints1, image2, keypoints2, miBins = 256, minPatchSize = 9):
    patches = _transform(image1, keypoints1, image2, keypoints2, minPatchSize)
    
    if patches[0] == None:
        return 0.0, -1.0
    
    else:
        return _computeMutualInformation(patches[0], patches[1]), _computeMSE(patches[0], patches[1])
