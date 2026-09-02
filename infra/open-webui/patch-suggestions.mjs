import fs from 'node:fs';

const target = '/src/src/lib/components/chat/Suggestions.svelte';
let source = fs.readFileSync(target, 'utf8');

function replaceExact(oldText, newText, expectedCount = 1) {
  const count = source.split(oldText).length - 1;
  if (count !== expectedCount) {
    throw new Error(
      `${target}: expected ${expectedCount} occurrence(s) of ${JSON.stringify(oldText)}, found ${count}`
    );
  }
  source = source.split(oldText).join(newText);
}

replaceExact(
  'sortedPrompts = [...(suggestionPrompts ?? [])].sort(() => Math.random() - 0.5);',
  'sortedPrompts = [...(suggestionPrompts ?? [])];'
);
replaceExact('<div class="h-36 w-full">', '<div class="w-full">');
replaceExact(
  '<div role="list" class="max-h-36 overflow-auto scrollbar-none items-start {className}">',
  '<div role="list" class="items-start {className}">'
);
replaceExact('transition line-clamp-1', 'transition', 2);

if (!source.includes('sortedPrompts = [...(suggestionPrompts ?? [])];')) {
  throw new Error(`${target}: ordered-suggestion marker missing after patch`);
}
if (source.includes('Math.random() - 0.5')) {
  throw new Error(`${target}: random suggestion ordering remains after patch`);
}

fs.writeFileSync(target, source);
console.log('OPENWEBUI_ORDERED_SUGGESTIONS_PATCH_PASS');
