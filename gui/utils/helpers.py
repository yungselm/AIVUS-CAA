from loguru import logger

def connect_consecutive_frames(missing: list) -> str:
        nums = sorted(set(missing))
        connected = []
        i = 0
        while i < len(nums):
            j = i
            while j < len(nums) - 1 and nums[j + 1] - nums[j] == 1:
                j += 1
            if i == j:
                connected.append([nums[i]])
            else:
                connected.append(nums[i : j + 1])
            i = j + 1
        connected = [
            (f'{sublist[0]}-{sublist[-1]}' if len(sublist) > 2 else ", ".join(map(str, sublist)))
            for sublist in connected
        ]
        return ", ".join(connected)