# Fix Row Level Security (RLS) for Test Table

## Quick Fix - Disable RLS (for development/hackathon)

Go to Supabase SQL Editor and run:

```sql
-- Disable RLS on test table
ALTER TABLE test DISABLE ROW LEVEL SECURITY;
```

## OR - Enable RLS with Public Access Policies

If you want to keep RLS enabled but allow public access:

```sql
-- Enable RLS
ALTER TABLE test ENABLE ROW LEVEL SECURITY;

-- Allow anyone to SELECT
CREATE POLICY "Enable read access for all users" ON test
  FOR SELECT USING (true);

-- Allow anyone to INSERT
CREATE POLICY "Enable insert access for all users" ON test
  FOR INSERT WITH CHECK (true);

-- Allow anyone to UPDATE
CREATE POLICY "Enable update access for all users" ON test
  FOR UPDATE USING (true);

-- Allow anyone to DELETE
CREATE POLICY "Enable delete access for all users" ON test
  FOR DELETE USING (true);
```

## Steps to Apply:

1. Go to: https://supabase.com/dashboard
2. Select your project
3. Click "SQL Editor" in sidebar
4. Copy and paste one of the SQL blocks above
5. Click "Run"
6. Test your API again

## For Production (Authenticated Users Only):

```sql
-- Enable RLS
ALTER TABLE test ENABLE ROW LEVEL SECURITY;

-- Only authenticated users can access
CREATE POLICY "Authenticated users can do everything" ON test
  FOR ALL USING (auth.uid() IS NOT NULL);
```
