import { Directive, computed, input } from '@angular/core';
import { hlm } from '@spartan-ng/ui-core';
import type { ClassValue } from 'clsx';

@Directive({
  selector: '[hlmProgress],brn-progress[hlm]',
  standalone: true,
  host: {
    '[class]': '_computedClass()',
  },
})
export class HlmProgressDirective {
  public readonly userClass = input<ClassValue>('', { alias: 'class' });
  protected _computedClass = computed(() =>
    hlm(
      'inline-flex relative h-4 w-full overflow-hidden rounded-full border',
      this.userClass()
    )
  );
}
