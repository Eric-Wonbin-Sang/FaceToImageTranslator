
def group_list_by_size(trend_list, list_size):
    # This groups any one dimensional list and groups it into lists with length of up to list_size
    if not trend_list:
        return []
    if len(trend_list) < list_size:
        return [trend_list]
    data_list_list = list([list(x) for x in zip(*[iter(trend_list)] * list_size)])
    if len(trend_list) % list_size != 0:
        data_list_list.append(trend_list[-(len(trend_list) % list_size):])
    return data_list_list
