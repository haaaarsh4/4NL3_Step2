import os, sys, time, json
import pandas as pd

input_dir = '/app/input_data/'
output_dir = '/app/output/'
sys.path.append('/app/program')
sys.path.append('/app/ingested_program')

def main():
    from model import Model

    train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(input_dir, 'test.csv'))

    train_df['dialogue'] = train_df['dialogue'].fillna('')
    test_df['dialogue'] = test_df['dialogue'].fillna('')

    X_train = train_df['dialogue'].tolist()
    y_train = train_df['name'].tolist()
    X_test = test_df['dialogue'].tolist()

    start = time.time()
    m = Model()
    m.fit(X_train, y_train)
    prediction = m.predict(X_test)
    duration = time.time() - start

    with open(os.path.join(output_dir, 'prediction'), 'w') as f:
        f.write('\n'.join(str(p) for p in prediction))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({'duration': duration}, f)

if __name__ == '__main__':
    main()
