import datetime
import json

import requests

fd = open('power_data.csv', 'a')

id_range = 1
current_data = []
for id in range(id_range):
    url = 'http://coosa-socket.herokuapp.com/power/?id=' + str(id)
    my_response = requests.get(url)
    if my_response.ok:
        json_data = json.loads(my_response.content)
        current_data.append(json_data['Power'])

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data_string = now + ", " + ", ".join(map(str, current_data)) + "\n"
print(data_string)

fd.write(data_string)
fd.close()
