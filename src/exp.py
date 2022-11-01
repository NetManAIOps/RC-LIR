from models import Data, Params, Result

if __name__ == '__main__':
    params = Params()
    result = Result(params)

    if params.save_res:
        save_file = open(params.save_res, 'w')

    count = 0
    try:
        for id in params.test_range:
            data = Data(id, params)
            rank = params.ranker.rank(data, params)
            for filter in params.filters:
                rank = filter.filter(data, rank)
            result.count(id, rank)

            count += 1
            print(f'{count} / {len(params.test_range)}')
            if params.print_res:
                rank.print()
            if params.save_res:
                save_file.write(str(id)+','+','.join(rank.sorted_fields())+'\n')
    except:
        result.print()

    if params.save_res:
        save_file.close()

    result.print()
