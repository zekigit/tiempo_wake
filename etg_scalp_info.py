study_path = '/Volumes/FAT/Time/ETG_scalp/'
data_path = study_path + 'bdf/'
log_path = '/Users/lpen/Documents/Experimentos/Drowsy Time/TimeGeneralization/analisis/scalp_behaviour/logs'

n_jobs = 2

sujetos = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '20', '21', '22']


sesiones = {'1': ('1', '2'),
            '2': ('1', '2', '3'),
            '3': ('1', '2', '3'),
            '4': ('3', ),
            '5': ('1', '2', '3'),
            '6': ('1', '3'),
            '7': ('1', '2', '3'),
            '8': ('1', '2', '3'),
            '9': ('1', '2', '3'),
            '10': ('1', '2', '3'),
            '11': ('2', ),
            '12': ('1', '2', '3'),
            '13': ('1', '2', '3'),
            '14': ('1', '2', '3'),
            '15': ('1', '2', '3'),
            '16': ('3',),
            '17': ('3', ),
            '18': ('1', '2', '3'),
            # '19': ('1', ),
            '20': ('1', '2', '3'),
            '21': ('1', '2', '3'),
            '22': ('2', '3')
            }


bad_channs = {'1': ['A1', 'A12', 'A18', 'A25', 'A26', 'A30', 'B9', 'C24', 'C25', 'C32', 'D1', 'D6'],
              '2': ['A1', 'A4', 'A9', 'A16', 'A25', 'A26', 'B7', 'B8', 'B21', 'B32', 'D8', 'D1', 'D14', 'D22', 'D23', 'D31'],
              '3': ['A1', 'A25', 'A26', 'B9', 'B10', 'C8', 'C9', 'C11', 'C30', 'D1'],
              '4': ['A1', 'A25', 'A26', 'A27', 'A28', 'B14', 'D1', 'D3', 'D17', 'D24'],
              '5': ['B1', 'B8', 'D12', 'D22', 'D32'],
              '6': ['A10', 'B1'],
              '7': ['C1'],
              '8': ['B1', 'C1'],
              '9': ['A1', 'A30', 'D8', 'D10'],
              '10': ['A1', 'A32', 'C7', 'C29', 'C32'],
              '11': ['A1'],
              '12': ['A1', 'A19', 'C5'],
              '13': ['A12', 'A23', 'B1', 'B10', 'B11', 'B26', 'D23'],
              '14': ['A29', 'C1'],
              '15': ['A1', 'B1', 'C1', 'D23'],
              '16': ['A20', 'B7', 'B23', 'C9', 'C31'],
              '17': ['C14', 'D24'],
              '18': ['D8', 'B1', 'C1'],
              '19': ['C1'],
              '20': ['A13', 'C1'],
              '21': ['A14', 'B1'],
              '22': ['B1', 'B19']
              }

ch_to_remove = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

marks = {'s1_short': 1, 's2_short': 2, 'exp_short_smaller': 7, 'exp_short_equal': 8, 'exp_short_bigger': 9,
         's1_long': 10, 's2_long': 20, 'exp_long_smaller': 70, 'exp_long_equal': 80, 'exp_long_bigger': 90}