"""
This script grabs a subset of the full public use QWI data set available at:

    http://download.vrdc.cornell.edu/qwipu/.

Specifically this script downloads the following QWI data for each U.S. state:

1. 'se': Sex by Education
2. 'f': unconditional on firm size or age
3. 'gm': core-based statistical areas (i.e., metro/micro-politan)
4. 'n4': NAICS Industry groups (4-digit)
5. 'oslp': State, local, and private ownership categories (BLS codes: 2,3,5;
    QWI code A00)
6. 'u': Note seasonally adjusted

See QWI public use data schema for more information about available data calls:

    http://download.vrdc.cornell.edu/qwipu/QWIPU_Data_Schema.pdf

Notes
-----

Seems like there QWI data are firm level data.

There are multiple file formats for downloading this data depending on the
time span of data desired.

"""
import requests
import us


def _get_base_url(year, quarter):
    """Return the base url for the QWI data for given year and quarter."""
    host = 'http://download.vrdc.cornell.edu'

    # checks whether or not base url is in the data archive
    if year < 2012:
        base_path = '/qwtpu/archive'
    else:
        base_path = '/qwtpu'

    path = base_path + '/R' + str(year) + 'Q' + str(quarter)
    base_url = host + path
    return base_url


def _get_file_name(st, char, fas, geo, ind, own, sa):
    """Given some query parameters, returns a file name."""
    file_name = ('qwi_' + st.lower() + '_' + char + '_' + fas + '_' + geo +
                 '_' + ind + '_' + own + '_' + sa + '.csv.gz')
    return file_name


for state in us.states.STATES:

    # extract the abbreviation for the state...
    tmp_abbr = state.abbr

    # ...construct the url ...
    tmp_base_url = _get_base_url(2013, 1)
    tmp_file_name = _get_file_name(tmp_abbr, 'se', 'f', 'gm', 'n4', 'oslp', 'u')
    tmp_url = tmp_base_url + tmp_file_name

    # ...make the connection and grab the zipped files...
    tmp_buffer = requests.get(tmp_url)

    # ...save them to disk...
    with open(tmp_file_name, 'wb') as tmp_zip_file:
        tmp_zip_file.write(tmp_buffer.content)

    print('Done with files for ' + tmp_abbr + '!')
    