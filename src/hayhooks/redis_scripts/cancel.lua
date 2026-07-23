-- hayhooks:cancel
local payload = redis.call('GET', KEYS[1])
if not payload then return 0 end
local record = cjson.decode(payload)
if record['status'] == 'completed' or record['status'] == 'failed' or record['status'] == 'canceled' then return 0 end
if record['cancel_requested_at'] ~= nil and record['cancel_requested_at'] ~= cjson.null then return 1 end
record['cancel_requested_at'] = ARGV[1]
record['cancel_reason'] = ARGV[2] ~= '' and ARGV[2] or cjson.null
record['sequence'] = (record['sequence'] or 0) + 1
record['updated_at'] = ARGV[1]
local progress = record['bounded_progress'] or {}
local next_sequence = #progress > 0 and ((progress[#progress]['sequence'] or 0) + 1) or 1
table.insert(progress, {
    sequence = next_sequence,
    kind = 'cancellation_requested',
    message = 'Cancellation requested',
    timestamp = ARGV[1],
    metadata = {}
})
local limit = tonumber(ARGV[4])
while #progress > limit do table.remove(progress, 1) end
record['bounded_progress'] = progress
if record['status'] == 'waiting' then
    local old_status = record['status']
    record['status'] = 'canceled'
    record['wait'] = cjson.null
    record['error'] = cjson.null
    record['result'] = cjson.null
    record['retry_at'] = cjson.null
    local remaining = redis.call('HINCRBY', KEYS[2], old_status, -1)
    if remaining < 0 then redis.call('HSET', KEYS[2], old_status, 0) end
    redis.call('HINCRBY', KEYS[2], 'canceled', 1)
    redis.call('ZADD', KEYS[3], ARGV[6], ARGV[5])
    redis.call('HSET', KEYS[4], ARGV[5], 'canceled')
    redis.call('SET', KEYS[1], cjson.encode(record), 'EX', ARGV[3])
    return 2
end
redis.call('SET', KEYS[1], cjson.encode(record))
return 1
