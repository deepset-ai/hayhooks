-- hayhooks:retire_incompatible
local payload = redis.call('GET', KEYS[1])
if not payload then
    redis.call('HDEL', KEYS[6], ARGV[6])
    redis.call('HDEL', KEYS[7], ARGV[6])
    return 0
end
local record = cjson.decode(payload)
local status = record['status']
if status == 'completed' or status == 'failed' or status == 'canceled' then
    redis.call('HDEL', KEYS[6], ARGV[6])
    return 0
end
if record['definition_revision'] == ARGV[1] or status == 'running' then return 0 end

record['result'] = cjson.null
record['retry_at'] = cjson.null
record['wait'] = cjson.null
local canceled = record['cancel_requested_at'] ~= nil and record['cancel_requested_at'] ~= cjson.null
if canceled then
    record['status'] = 'canceled'
    record['error'] = cjson.null
else
    record['status'] = 'failed'
    record['error'] = {
        type = 'DefinitionRevisionConflictError',
        message = 'Execution cannot continue after its deployment definition was replaced',
        retryable = false,
        code = 'definition_revision_conflict'
    }
end
record['sequence'] = (record['sequence'] or 0) + 1
record['updated_at'] = ARGV[2]

local progress = record['bounded_progress'] or {}
local next_sequence = #progress > 0 and ((progress[#progress]['sequence'] or 0) + 1) or 1
table.insert(progress, {
    sequence = next_sequence,
    kind = canceled and 'canceled' or 'definition_revision_conflict',
    message = canceled
        and 'Execution canceled during deployment replacement'
        or 'Execution retired because its deployment definition was replaced',
    timestamp = ARGV[2],
    metadata = {}
})
local limit = tonumber(ARGV[4])
while #progress > limit do table.remove(progress, 1) end
record['bounded_progress'] = progress

local remaining = redis.call('HINCRBY', KEYS[3], status, -1)
if remaining < 0 then redis.call('HSET', KEYS[3], status, 0) end
redis.call('HINCRBY', KEYS[3], record['status'], 1)
redis.call('ZADD', KEYS[4], ARGV[5], record['execution_id'])
redis.call('HSET', KEYS[5], record['execution_id'], record['status'])
redis.call('SET', KEYS[1], cjson.encode(record), 'EX', ARGV[3])
redis.call('ZREM', KEYS[2], record['execution_id'])
redis.call('HDEL', KEYS[6], record['execution_id'])
redis.call('HSET', KEYS[7], record['execution_id'], record['sequence'] or 0)
return 1
