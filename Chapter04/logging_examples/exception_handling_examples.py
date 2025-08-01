import logging

def process(data_to_be_processed):
    '''Dummy example that returns original data plus 1'''
    return data_to_be_processed + 1

def process_data(data): 
    try: 
        # Do some processing on the data 
        result = process(data) 
    except Exception as e: 
        # Log the exception 
        logging.exception("Exception occurred while processing data") 

        # Re-raise the exception with a new exception
        new_exception = ValueError("Error processing data") 
        raise new_exception from e

    return result 


